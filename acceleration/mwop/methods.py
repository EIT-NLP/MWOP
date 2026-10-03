import torch
from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import _flash_attn_forward
from ops.compiled_ops import compiled_rmsnorm, compiled_rope_qk
METHODS = ('zoo', 'pdrop', 'shortv', 'redundancy_lens')

@torch.compile(fullgraph=True)
def merge_attention(a, la, b, lb):
    weight = torch.sigmoid(la - lb).transpose(1, 2).unsqueeze(-1)
    return (a.float() * weight + b.float() * (1 - weight)).to(a.dtype)

class HollowAttention(torch.nn.Module):

    def __init__(self, span, window=256):
        super().__init__()
        self.span = tuple(span)
        self.window = window

    def forward(self, q, k, v, **kwargs):
        p, e = self.span
        prefix = flash_attn_func(q[:, :p], k[:, :p], v[:, :p], causal=True)
        suffix = flash_attn_func(q[:, e:], k, v, causal=True)
        a, la, _, _ = _flash_attn_forward(q[:, p:e], k[:, :p], v[:, :p], 0.0, q.shape[-1] ** (-0.5), False, -1, -1, 0.0, None, False)
        b, lb, _, _ = _flash_attn_forward(q[:, p:e], k[:, p:e], v[:, p:e], 0.0, q.shape[-1] ** (-0.5), True, self.window, 0, 0.0, None, False)
        visual = merge_attention(a, la, b, lb)
        return torch.cat((prefix, visual, suffix), dim=1)

class TextQueryAttention(torch.nn.Module):

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def forward(self, q, k, v, **kwargs):
        p = self.prefix
        prefix = flash_attn_func(q[:, :p], k[:, :p], v[:, :p], causal=True)
        suffix = flash_attn_func(q[:, p:], k, v, causal=True)
        return torch.cat((prefix, suffix), dim=1)

class ShortVAttention(torch.nn.Module):

    def __init__(self, original, span):
        super().__init__()
        self.original = original
        self.span = tuple(span)
        self.attention_impl = TextQueryAttention(span[0])

    def forward(self, hidden_states, position_embeddings=None, past_key_values=None, cache_position=None, **kwargs):
        a = self.original
        p, e = self.span
        norm = compiled_rmsnorm(hidden_states, a.input_layernorm.weight, a.input_layernorm.variance_epsilon)
        tx = torch.cat((norm[:, :p], norm[:, e:]), dim=1)
        q = a.q_proj(tx).view(*tx.shape[:2], -1, a.head_dim)
        k = a.k_proj(norm).view(*norm.shape[:2], -1, a.head_dim)
        v = a.v_proj(norm).view(*norm.shape[:2], -1, a.head_dim)
        if a.q_norm:
            q = compiled_rmsnorm(q, a.q_norm.weight, a.q_norm.variance_epsilon)
        if a.k_norm:
            k = compiled_rmsnorm(k, a.k_norm.weight, a.k_norm.variance_epsilon)
        cos, sin = position_embeddings
        tc, ts = [torch.cat((z[:, :p], z[:, e:]), dim=1) for z in (cos, sin)]
        q = compiled_rope_qk(q, q, tc, ts)[0]
        k = compiled_rope_qk(k, k, cos, sin)[0]
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, a.layer_idx, dict(sin=sin, cos=cos, cache_position=cache_position))
        y = self.attention_impl(q=q, k=k, v=v)
        y = a.o_proj(y.flatten(-2))
        residual = torch.cat((hidden_states[:, :p], hidden_states[:, e:]), dim=1)
        y = y + residual
        out = hidden_states.clone()
        out[:, :p] = y[:, :p]
        out[:, e:] = y[:, p:]
        return out

class NoVisualFFN(torch.nn.Module):

    def __init__(self, original, span):
        super().__init__()
        self.original = original
        self.span = tuple(span)

    def forward(self, hidden_states, **kwargs):
        p, e = self.span
        f = self.original
        text = torch.cat((hidden_states[:, :p], hidden_states[:, e:]), dim=1)
        x = compiled_rmsnorm(text, f.post_attention_layernorm.weight, f.post_attention_layernorm.variance_epsilon)
        y = f._text(x) + text
        out = hidden_states.clone()
        out[:, :p] = y[:, :p]
        out[:, e:] = y[:, p:]
        return out

class MethodController:

    def __init__(self, model, config):
        self.model, self.config = (model, config)
        self.prefix, self.visual, self.suffix = [config[k] for k in ('prefix', 'visual', 'suffix')]
        self.span = (self.prefix, self.prefix + self.visual)
        self.original_attention = [l.self_attn for l in model.model.layers]
        self.original_ffn = [l.mlp for l in model.model.layers]
        self.short_attention = [ShortVAttention(a, self.span) for a in self.original_attention]
        self.short_ffn = [NoVisualFFN(f, self.span) for f in self.original_ffn]
        self.hollow = [HollowAttention(self.span, config['rl_preceding_visual_window']) for _ in model.model.layers]
        self.state = {}
        self.handles = []
        self.method = None
        self.active = False
        self.handles.append(model.model.register_forward_pre_hook(self.start, with_kwargs=True))
        for i, layer in enumerate(model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(self.before(i), with_kwargs=True))

    def configure(self, method):
        assert method in METHODS
        self.method = method
        self.active = True
        cfg = self.config
        p, v, s = (self.prefix, self.visual, self.suffix)
        for l, a, f in zip(self.model.model.layers, self.original_attention, self.original_ffn):
            l.self_attn = a
            l.mlp = f
        self.model.configure('dense')
        self.counts = [v] * 28
        if method == 'zoo':
            self.counts = [cfg['zoo_keep_visual']] * 28
        if method == 'pdrop':
            for boundary, count in zip(cfg['pdrop_boundaries'], cfg['pdrop_visual']):
                self.counts[boundary:] = [count] * (28 - boundary)
        self.positions = []
        self.transitions = {}
        previous = v
        original = list(range(p + v + s))
        device = self.model.model.embed_tokens.weight.device
        for i, count in enumerate(self.counts):
            if count != previous:
                idx = list(range(p)) + [p + j * previous // count for j in range(count)] + list(range(p + previous, p + previous + s))
                self.transitions[i] = torch.tensor(idx, device=device, dtype=torch.long)
                original = [original[j] for j in idx]
            self.positions.append(list(original))
            previous = count
        self.lengths = [p + c + s for c in self.counts]
        for i, l in enumerate(self.model.model.layers):
            if method == 'shortv' and i in cfg['shortv_layers']:
                l.self_attn = self.short_attention[i]
                l.mlp = self.short_ffn[i]
            if method == 'redundancy_lens':
                if i in cfg['rl_attention_layers']:
                    l.self_attn.attention_impl = self.hollow[i]
                if i in cfg['rl_ffn_layers']:
                    l.mlp.mode = 'packed'

    def changed(self, layer, component):
        if self.method in ('zoo', 'pdrop'):
            return self.counts[layer] != self.visual
        if self.method == 'shortv':
            return layer in self.config['shortv_layers']
        return layer in self.config['rl_ffn_layers' if component == 'ffn_module' else 'rl_attention_layers']

    def start(self, module, args, kwargs):
        self.state.clear()
        if self.active and self.method == 'zoo':
            idx = self.transitions[0]
            kwargs = dict(kwargs)
            kwargs['inputs_embeds'] = kwargs['inputs_embeds'].index_select(1, idx)
            kwargs['attention_mask'] = kwargs['attention_mask'].index_select(1, idx)
            kwargs['position_ids'] = idx[None]
        return (args, kwargs)

    def before(self, i):

        def hook(module, args, kwargs):
            if not self.active or self.method != 'pdrop':
                return (args, kwargs)
            kwargs = dict(kwargs)
            if i == 0:
                self.state.update(pe=kwargs['position_embeddings'], pos=kwargs['position_ids'], cache=kwargs['cache_position'])
            if i in self.transitions:
                idx = self.transitions[i]
                args = (args[0].index_select(1, idx), *args[1:])
                self.state['pe'] = tuple((z.index_select(1, idx) for z in self.state['pe']))
                self.state['pos'] = self.state['pos'].index_select(-1, idx)
                self.state['cache'] = self.state['cache'].index_select(0, idx)
            kwargs.update(position_embeddings=self.state['pe'], position_ids=self.state['pos'], cache_position=self.state['cache'])
            return (args, kwargs)
        return hook
