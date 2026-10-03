from types import SimpleNamespace
import torch
from transformers import AutoConfig, LlavaOnevisionForConditionalGeneration
from wrapper.static.dense import DenseForCausalLM
from .attention import RegionAttention
from .config import flags_tensor
from .ffn import VisionMLP

class MWOPForCausalLM(DenseForCausalLM):
    _supports_flash_attn = True

    def __init__(self, config, block, plans, length, span, alignment=64, tiles=None):
        pruning = dict(cache_type='base', attention=dict(pruning_type='base', estimated_sparsity=0))
        super().__init__(config, pruning, block)
        self.length, self.span = (length, tuple(span))
        for i, layer in enumerate(self.model.layers):
            layer.mlp = VisionMLP(config, pruning, layer.mlp, layer.post_attention_layernorm, plans[i]['keep'], span, alignment)
            attn = layer.self_attn
            attn.mwop_dense = attn.attention_impl
            attn.mwop_region = RegionAttention(flags_tensor(plans[i], attn.q_proj.weight.device), span, length, tile=(tiles or {}).get(str(i), (128, 32, 4, 2)))
        self.eval()

    def configure(self, mode='both', backend='triton', reference_ffn=False):
        assert mode in ('dense', 'attention_only', 'ffn_only', 'both')
        for layer in self.model.layers:
            a = layer.self_attn
            a.mwop_region.backend = backend
            if mode in ('both', 'attention_only'):
                if backend == 'flex' and a.mwop_region.block is None:
                    a.mwop_region.prepare_flex()
                a.attention_impl = a.mwop_region
            else:
                a.attention_impl = a.mwop_dense
            layer.mlp.mode = ('reference' if reference_ffn else 'packed') if mode in ('both', 'ffn_only') else 'dense'

    def forward(self, *args, **kwargs):
        x = kwargs.get('inputs_embeds')
        if x is None or x.shape[:2] != (1, self.length) or kwargs.get('past_key_values') is not None:
            raise ValueError('MWOP runner requires B=1, fixed-length inputs_embeds and a fresh cache')
        return super().forward(*args, **kwargs)

def load_model(path, plans, length, span, tiles=None, alignment=64):
    outer_config = AutoConfig.from_pretrained(path, local_files_only=True)
    if outer_config.model_type != 'llava_onevision':
        raise ValueError('The acceleration runner requires a Hugging Face llava_onevision checkpoint, not a native llava_qwen training checkpoint')
    cls = LlavaOnevisionForConditionalGeneration
    outer = cls.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True, attn_implementation='flash_attention_2')
    config = outer.config.text_config
    assert (config.num_hidden_layers, config.hidden_size, config.intermediate_size, config.num_attention_heads, config.num_key_value_heads) == (28, 3584, 18944, 28, 4)
    if hasattr(outer, 'language_model') and hasattr(outer.language_model, 'lm_head'):
        language = outer.language_model
        block = SimpleNamespace(model=language.model, lm_head=language.lm_head)
    else:
        language = outer.model.language_model
        block = SimpleNamespace(model=language, lm_head=outer.lm_head)
    block.model.to('cuda')
    block.lm_head.to('cuda')
    model = MWOPForCausalLM(config, block, plans, length, span, alignment, tiles)
    return model
