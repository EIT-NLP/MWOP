import torch
import torch.nn as nn
import nvtx
from typing import Optional, Tuple, Dict, List, Union, Any
from einops import rearrange
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput
from torch.profiler import record_function
from ops import __ATTENTION__, __SPARSE_ATTENTION__, __THRESHOLD__, __MLP__, __KV_CACHE__, __APPROXIMATOR__
from ops.compiled_ops import compiled_rmsnorm, compiled_rope_qk
from wrapper.base import PrunedMLP, PrunedAttention, PrunedDecoderLayer, PrunedModelForCausalLM

class DenseMLP(PrunedMLP):

    def __init__(self, config: PretrainedConfig, pruning_config: Dict[str, Any], block: nn.Module, post_attention_layernorm: nn.Module, **kwargs):
        super().__init__(config, pruning_config, block, **kwargs)
        self._support_pruning_components = []
        self.mlp_impl = __MLP__['base']()
        self.post_attention_layernorm = post_attention_layernorm

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = hidden_states
        hidden_states = compiled_rmsnorm(hidden_states, self.post_attention_layernorm.weight, self.post_attention_layernorm.variance_epsilon)
        with nvtx.annotate('glu', color='green'):
            with record_function('glu'):
                glu_output = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        with nvtx.annotate('down_proj', color='green'):
            with record_function('down_proj'):
                ffn_output = self.down_proj(glu_output)
        return ffn_output + residual

class DenseAttention(PrunedAttention):

    def __init__(self, config: PretrainedConfig, pruning_config: Dict[str, Any], block: nn.Module, input_layernorm: nn.Module, **kwargs):
        super().__init__(config, pruning_config, block, **kwargs)
        self._support_pruning_components = ['attention.']
        for key in self._support_pruning_components:
            prefix, suffix = key.split('.')
            if suffix == '':
                setattr(self, f'{prefix}_kwargs', self.pruning_config.get(prefix, {}))
            else:
                setattr(self, f'{suffix}_kwargs', self.pruning_config[prefix].get(suffix, {}))
        self.attention_impl = __SPARSE_ATTENTION__[self.attention_kwargs.get('pruning_type', 'base')]()
        self.threshold_impl = __THRESHOLD__[self.attention_kwargs.get('pruning_type', 'base')](**self.attention_kwargs)
        self.input_layernorm = input_layernorm

    @nvtx.annotate('Attention', color='blue')
    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Cache]=None, cache_position: Optional[torch.LongTensor]=None, pad_offset: Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
        residual = hidden_states
        hidden_states = compiled_rmsnorm(hidden_states, self.input_layernorm.weight, self.input_layernorm.variance_epsilon)
        with record_function('qkv_proj'):
            with nvtx.annotate('q_proj', color='blue'):
                q = self.q_proj(hidden_states)
            with nvtx.annotate('k_proj', color='blue'):
                k = self.k_proj(hidden_states)
            with nvtx.annotate('v_proj', color='blue'):
                v = self.v_proj(hidden_states)
        q, k, v = list(map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_dim), [q, k, v]))
        if self.q_norm:
            q = compiled_rmsnorm(q, self.q_norm.weight, self.q_norm.variance_epsilon)
        if self.k_norm:
            k = compiled_rmsnorm(k, self.k_norm.weight, self.k_norm.variance_epsilon)
        cos, sin = position_embeddings
        q, k = compiled_rope_qk(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position, 'inplace_update_kvcache': kwargs.get('inplace_update_kvcache', False)}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        input_kwargs = dict(q=q, k=k, v=v, attention_mask=attention_mask, pad_offset=pad_offset, execute_block=getattr(self, 'execute_block', None), prefill_impl=self.attention_kwargs.get('pruning_type'), decode_impl=self.attention_kwargs.get('pruning_type'), backend=self.attention_kwargs.get('backend', 'triton'))
        input_kwargs.update(self.threshold_impl.get_threshold_kwargs())
        with nvtx.annotate('attention', color='blue'):
            with record_function('attention'):
                attn_output = self.attention_impl(**input_kwargs)
        attn_output = rearrange(attn_output, '... h d -> ... (h d)')
        with nvtx.annotate('o_proj', color='blue'):
            with record_function('o_proj'):
                attn_output = self.o_proj(attn_output)
        return attn_output + residual

class DenseDecoderLayer(PrunedDecoderLayer):

    def __init__(self, config: PretrainedConfig, pruning_config: Dict[str, Any], block: nn.Module, layer_idx: int, **kwargs):
        super().__init__(config, pruning_config, block, layer_idx, **kwargs)
        self._support_pruning_components = []
        self.self_attn = DenseAttention(config, pruning_config, block.self_attn, block.input_layernorm, **kwargs)
        self.mlp = DenseMLP(config, pruning_config, block.mlp, block.post_attention_layernorm, **kwargs)

    def generate_pruning_kwargs(self, batch_size: int, query_len: int, kv_cache_len: int, device: Optional[torch.device]='cuda:0', **kwargs) -> Dict[str, torch.Tensor]:
        mask = self.self_attn.threshold_impl.generate_mask(batch_size=batch_size, query_len=query_len, kv_cache_len=kv_cache_len, num_key_heads=self.config.num_key_value_heads, sparsity=self.self_attn.attention_kwargs.get('estimated_sparsity', 0), device=device)
        setattr(self.self_attn, 'execute_block', mask)
        return mask

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Cache]=None, use_cache: Optional[bool]=False, cache_position: Optional[torch.LongTensor]=None, position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]]=None, pad_offset: Optional[torch.Tensor]=None, **kwargs):
        with nvtx.annotate(f'Layer_{self.layer_idx}', color='red'):
            hidden_states = self.self_attn(hidden_states, attention_mask=None, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, pad_offset=pad_offset, **kwargs)
            hidden_states = self.mlp(hidden_states)
            return hidden_states

class DensePretrainedModel(PreTrainedModel):
    config: PretrainedConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = False
    _no_split_modules = ['DenseDecoderLayer']
    _skip_keys_device_placement = ['past_key_values']
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = True
    _supports_attention_backend = False
    _can_record_outputs = {'hidden_states': DenseDecoderLayer, 'attentions': DenseAttention}

class DenseModel(DensePretrainedModel):

    def __init__(self, config: PretrainedConfig, pruning_config: Dict[str, Any], block: PreTrainedModel, **kwargs):
        super().__init__(config)
        self.pruning_config = pruning_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = block.embed_tokens
        self.layers = nn.ModuleList([DenseDecoderLayer(config, pruning_config, block.layers[i], i) for i in range(config.num_hidden_layers)])
        self.norm = block.norm
        self.rotary_emb = block.rotary_emb

    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Cache]=None, inputs_embeds: Optional[torch.FloatTensor]=None, cache_position: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, **kwargs) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You must specify exactly one of input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = __KV_CACHE__[self.pruning_config.get('cache_type', 'base')](self.config)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        pad_offset = attention_mask.shape[1] - attention_mask.sum(-1)
        for i, decoder_layer in enumerate(self.layers[:self.config.num_hidden_layers]):
            hidden_states = decoder_layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, cache_position=cache_position, position_embeddings=position_embeddings, pad_offset=pad_offset, **kwargs)
        hidden_states = compiled_rmsnorm(hidden_states, self.norm.weight, self.norm.variance_epsilon)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

class DenseForCausalLM(PrunedModelForCausalLM):

    def __init__(self, config: PretrainedConfig, pruning_config: Dict[str, Any], block: PreTrainedModel, **kwargs):
        super().__init__(config, pruning_config, block, **kwargs)
        self.model = DenseModel(config, pruning_config, block.model, **kwargs)

    def generate_pruning_kwargs(self, **kwargs) -> Dict[str, torch.Tensor]:
        for layer in self.model.layers:
            layer.generate_pruning_kwargs(**kwargs)
        return {}

    def post_load(self, router_ckpt: Optional[Dict[str, torch.Tensor]]=None, lora_ckpt: Optional[Dict[str, torch.Tensor]]=None, full_ckpt: Optional[Dict[str, torch.Tensor]]=None, lora_rank: Optional[int]=16, lora_alpha: Optional[float]=32, **kwargs):
        pass
