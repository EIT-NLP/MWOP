import torch
import torch.nn.functional as F
from ops.compiled_ops import compiled_rmsnorm
from wrapper.static.dense import DenseMLP

class VisionMLP(DenseMLP):

    def __init__(self, config, pruning_config, block, norm, keep, span, alignment=64):
        super().__init__(config, pruning_config, block, norm)
        self.span = tuple(span)
        self.mode = 'dense'
        device = self.gate_proj.weight.device
        indices = torch.tensor(keep, device=device, dtype=torch.long)
        self.kept = len(keep)
        self.width = min(self.intermediate_size, (self.kept + alignment - 1) // alignment * alignment)
        self.register_buffer('keep', indices)
        mask = self.gate_proj.weight.new_zeros(self.intermediate_size)
        mask[indices] = 1
        self.register_buffer('keep_mask', mask)
        with torch.no_grad():
            for name in ('gate', 'up'):
                proj = getattr(self, name + '_proj')
                w = proj.weight.new_zeros((self.width, self.hidden_size))
                w[:self.kept].copy_(proj.weight.index_select(0, indices))
                self.register_buffer('vision_' + name, w)
                bias = None
                if proj.bias is not None:
                    bias = proj.bias.new_zeros(self.width)
                    bias[:self.kept].copy_(proj.bias.index_select(0, indices))
                self.register_buffer('vision_' + name + '_bias', bias)
            w = self.down_proj.weight.new_zeros((self.hidden_size, self.width))
            w[:, :self.kept].copy_(self.down_proj.weight.index_select(1, indices))
            self.register_buffer('vision_down', w)

    def _text(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, hidden_states, **kwargs):
        if self.mode == 'dense' or self.kept == self.intermediate_size:
            return super().forward(hidden_states)
        residual = hidden_states
        x = compiled_rmsnorm(hidden_states, self.post_attention_layernorm.weight, self.post_attention_layernorm.variance_epsilon)
        s, e = self.span
        if self.mode == 'reference':
            h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            h[:, s:e] = h[:, s:e] * self.keep_mask
            return self.down_proj(h) + residual
        out = torch.empty_like(x)
        text_x = torch.cat((x[:, :s], x[:, e:]), dim=1)
        if text_x.shape[1]:
            text_out = self._text(text_x)
            out[:, :s] = text_out[:, :s]
            out[:, e:] = text_out[:, s:]
        if self.kept:
            visual_x = x[:, s:e]
            h = self.act_fn(F.linear(visual_x, self.vision_gate, self.vision_gate_bias))
            h = h * F.linear(visual_x, self.vision_up, self.vision_up_bias)
            out[:, s:e] = F.linear(h, self.vision_down, self.down_proj.bias)
        else:
            out[:, s:e] = 0 if self.down_proj.bias is None else self.down_proj.bias
        return out + residual
