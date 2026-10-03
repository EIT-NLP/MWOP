import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union, Any
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput

class PrunedMLP(nn.Module):

    def __init__(self, config: PretrainedConfig, pruning_config: Dict[str, Any], block: nn.Module, **kwargs):
        super().__init__()
        self._support_pruning_components = []
        self._applied_pruning = []
        self._leaf_modules = ('up_proj', 'gate_proj', 'down_proj')
        self.config = config
        self.pruning_config = pruning_config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.activation = config.hidden_act
        self.act_fn = block.act_fn
        self.up_proj = block.up_proj
        self.gate_proj = block.gate_proj
        self.down_proj = block.down_proj

class PrunedAttention(nn.Module):

    def __init__(self, config: PretrainedConfig, pruning_config: Dict[str, Any], block: nn.Module, **kwargs):
        super().__init__()
        self._support_pruning_components = []
        self._applied_pruning = []
        self._leaf_modules = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
        self.config = config
        self.pruning_config = pruning_config
        self.layer_idx = block.layer_idx
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** (-0.5)
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = block.q_proj
        self.k_proj = block.k_proj
        self.v_proj = block.v_proj
        self.o_proj = block.o_proj
        self.q_norm = getattr(block, 'q_norm', None)
        self.k_norm = getattr(block, 'k_norm', None)

class PrunedDecoderLayer(nn.Module):

    def __init__(self, config: PretrainedConfig, pruning_config: Dict[str, Any], block: nn.Module, layer_idx: int, **kwargs):
        super().__init__()
        self._support_pruning_components = []
        self._applied_pruning = []
        self._leaf_modules = ('self_attn', 'mlp')
        self.config = config
        self.pruning_config = pruning_config
        self.layer_idx = layer_idx
        self.input_layernorm = block.input_layernorm
        self.post_attention_layernorm = block.post_attention_layernorm
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_attention_heads

class PrunedModelForCausalLM(PreTrainedModel, GenerationMixin):

    def __init__(self, config: PretrainedConfig, pruning_config: Dict[str, Any], block: PreTrainedModel, **kwargs):
        super().__init__(config)
        self.pruning_config = pruning_config
        self.vocab_size = config.vocab_size
        self.lm_head = block.lm_head

    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Cache]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, cache_position: Optional[torch.LongTensor]=None, logits_to_keep: Union[int, torch.Tensor]=0, estimated_sparsity: Optional[float]=0.0, **kwargs) -> CausalLMOutputWithPast:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, cache_position=cache_position, estimated_sparsity=estimated_sparsity, **kwargs)
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def generate_pruning_kwargs(self, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def post_load(self, **kwargs):
        raise NotImplementedError()
