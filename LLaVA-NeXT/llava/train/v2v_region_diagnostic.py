from __future__ import annotations
import os
from typing import Optional
import torch
try:
    from transformers import TrainerCallback as _TC
except Exception:
    _TC = object
try:
    from llava.constants import IMAGE_TOKEN_INDEX
except Exception:
    IMAGE_TOKEN_INDEX = -200

class V2VRegionDiagnosticCallback(_TC):

    def __init__(self, max_steps_to_log: int=5, log_path: Optional[str]=None):
        self.max_steps = int(max_steps_to_log)
        self.log_path = log_path
        self._fp = None
        self._hook_handle = None
        self._captured = {}

    def _log(self, msg: str) -> None:
        line = f'[v2v-diag] {msg}'
        try:
            from llava.utils import rank0_print
            rank0_print(line)
        except Exception:
            print(line, flush=True)
        if self._fp is not None:
            self._fp.write(line + '\n')
            self._fp.flush()

    def _is_rank0(self, args) -> bool:
        return getattr(args, 'local_rank', -1) in (-1, 0)

    @staticmethod
    def _first_self_attn(model):
        import re
        pat = re.compile('(?:^|\\.)model\\.layers\\.0\\.self_attn$')
        for name, mod in model.named_modules():
            if pat.search(name) and hasattr(mod, 'num_heads'):
                return (name, mod)
        pat2 = re.compile('(?:^|\\.)model\\.layers\\.\\d+\\.self_attn$')
        for name, mod in model.named_modules():
            if pat2.search(name) and hasattr(mod, 'num_heads'):
                return (name, mod)
        return (None, None)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._is_rank0(args) or model is None:
            return
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self._fp = open(self.log_path, 'w', encoding='utf-8')
        self._log('=' * 70)
        self._log('V2VRegionDiagnosticCallback ENABLED')
        self._log(f'  IMAGE_TOKEN_INDEX = {IMAGE_TOKEN_INDEX}')
        rt = getattr(model, '_ov_region_runtime', None)
        self._log(f'  _ov_region_runtime present: {rt is not None}')
        v2v_layers = []
        import re
        pat = re.compile('(?:^|\\.)model\\.layers\\.(\\d+)\\.self_attn$')
        for name, mod in model.named_modules():
            m = pat.search(name)
            if m and getattr(mod, '_ov_v2v_head_indices', None):
                v2v_layers.append((int(m.group(1)), len(mod._ov_v2v_head_indices)))
        self._log(f'  layers carrying _ov_v2v_head_indices: {len(v2v_layers)} (e.g. {v2v_layers[:3]})')
        name, mod = self._first_self_attn(model)
        if mod is None:
            self._log('  WARNING: could not find layers.0.self_attn to hook')
            return
        self._log(f'  hooking {name} forward to capture vis range')
        from llava.model.language_model.attention_mask_plugin import _compute_visual_range

        def pre_hook(module, fwd_args, fwd_kwargs):
            try:
                hs = fwd_kwargs.get('hidden_states')
                if hs is None and len(fwd_args) > 0:
                    hs = fwd_args[0]
                q_len = int(hs.shape[1]) if hs is not None else None
                rt = getattr(module, '_ov_region_runtime', None)
                kv = q_len
                rng = _compute_visual_range(rt or {}, kv) if rt and kv else None
                self._captured = {'q_len': q_len, 'kv_seq_len': kv, 'runtime': dict(rt) if rt else None, 'vis_range': rng, 'v2v_heads_here': tuple(getattr(module, '_ov_v2v_head_indices', ()) or ())[:5]}
            except Exception as e:
                self._captured = {'hook_error': repr(e)}
            return None
        try:
            self._hook_handle = mod.register_forward_pre_hook(pre_hook, with_kwargs=True)
        except TypeError:

            def pre_hook_pos(module, fwd_args):
                try:
                    hs = fwd_args[0] if fwd_args else None
                    q_len = int(hs.shape[1]) if hs is not None else None
                    rt = getattr(module, '_ov_region_runtime', None)
                    rng = _compute_visual_range(rt or {}, q_len) if rt and q_len else None
                    self._captured = {'q_len': q_len, 'kv_seq_len': q_len, 'runtime': dict(rt) if rt else None, 'vis_range': rng}
                except Exception as e:
                    self._captured = {'hook_error': repr(e)}
                return None
            self._hook_handle = mod.register_forward_pre_hook(pre_hook_pos)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self._is_rank0(args):
            return
        step = state.global_step
        if step > self.max_steps:
            return
        cap = self._captured
        self._log('-' * 70)
        self._log(f'step {step}')
        if not cap:
            self._log('  (no capture -- hook did not fire?)')
            return
        if 'hook_error' in cap:
            self._log(f"  hook error: {cap['hook_error']}")
            return
        rt = cap.get('runtime')
        kv = cap.get('kv_seq_len')
        rng = cap.get('vis_range')
        self._log(f'  kv_seq_len (q_len prefill) = {kv}')
        self._log(f'  runtime = {rt}')
        if rng is None:
            self._log('  vis_range = None  ==>  V2V MASK IS A NO-OP THIS STEP! (either has_image False, or range invalid). If this persists, mask_v2v never fires during training.')
        else:
            vs, ve = rng
            width = ve - vs
            self._log(f'  vis_range = [{vs}, {ve})   width = {width} visual tokens')
            if rt:
                n_ph = rt.get('n_img_placeholders')
                osl = rt.get('orig_seq_len')
                exp_img = kv - (osl - n_ph) if kv and osl and (n_ph is not None) else None
                self._log(f'    image placeholders in raw ids = {n_ph}, orig_seq_len = {osl}, expanded kv = {kv}')
                self._log(f"    => expected #image tokens (kv - text) = {exp_img}; mask width = {width}  {('MATCH' if exp_img == width else 'MISMATCH!! V2V region width != real image-token count')}")
                if ve > kv:
                    self._log('    ERROR: vis_e > kv_seq_len -> region runs past sequence!')
                if vs != n_ph_pos_guess(rt):
                    pass
        self._captured = {}

    def on_train_end(self, args, state, control, **kwargs):
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
        if self._fp is not None:
            self._fp.close()
            self._fp = None

def n_ph_pos_guess(rt):
    return rt.get('img_placeholder_pos', -1)
