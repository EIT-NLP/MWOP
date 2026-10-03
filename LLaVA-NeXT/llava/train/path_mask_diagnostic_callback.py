from __future__ import annotations
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
try:
    from transformers import TrainerCallback as _TrainerCallback
except Exception:
    _TrainerCallback = object
try:
    from llava.utils import rank0_print as _rank0_print
except Exception:

    def _rank0_print(*a, **k):
        print(*a, **k)
_LAYER_RE = re.compile('\\.layers\\.(\\d+)\\.self_attn$')

class PathMaskDiagnosticCallback(_TrainerCallback):

    def __init__(self, max_steps_to_log: int=3, config_path: Optional[str]=None):
        self.max_steps_to_log = max(1, int(max_steps_to_log))
        self.config_path = config_path
        self._steps_seen = 0
        self._model = None

    def _log(self, msg: str) -> None:
        _rank0_print(msg)

    @staticmethod
    def _expected_heads_from_config(path: Optional[str]) -> Dict[str, Set[Tuple[int, int]]]:
        out: Dict[str, Set[Tuple[int, int]]] = {'v2v': set(), 't2v': set(), 't2t': set()}
        if path is None or not os.path.exists(path):
            return out
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception as e:
            _rank0_print(f'[path-mask diag] WARN: failed to read {path}: {e}')
            return out

        def _accum(component: dict, base_dir: str) -> None:
            atype = component.get('ablation_type', '')
            heads = component.get('heads', [])
            if 'heads_from' in component:
                ref = component['heads_from']
                ref_full = ref if os.path.isabs(ref) else os.path.join(base_dir, ref)
                try:
                    with open(ref_full, 'r', encoding='utf-8') as f:
                        ref_cfg = json.load(f)
                    heads = list(ref_cfg.get('heads', [])) + list(heads)
                except Exception as e:
                    _rank0_print(f'[path-mask diag] WARN: heads_from {ref_full!r}: {e}')
            for item in heads:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                pair = (int(item[0]), int(item[1]))
                if atype == 'mask_v2v':
                    out['v2v'].add(pair)
                elif atype == 'mask_t2v':
                    out['t2v'].add(pair)
                elif atype == 'mask_t2t':
                    out['t2t'].add(pair)
        base_dir = os.path.dirname(os.path.abspath(path))
        if cfg.get('ablation_type') == 'composite':
            for c in cfg.get('components', []):
                _accum(c, base_dir)
        else:
            _accum(cfg, base_dir)
        return out

    @staticmethod
    def _attached_heads_from_model(model) -> Dict[str, Dict[int, Tuple[int, ...]]]:
        out: Dict[str, Dict[int, Tuple[int, ...]]] = {'v2v': {}, 't2v': {}, 't2t': {}}
        if model is None:
            return out
        for name, mod in model.named_modules():
            m = _LAYER_RE.search(name)
            if not m:
                continue
            layer_idx = int(m.group(1))
            for kind in ('v2v', 't2v', 't2t'):
                attr = f'_ov_{kind}_head_indices'
                vals = getattr(mod, attr, None)
                if vals:
                    out[kind][layer_idx] = tuple((int(h) for h in vals))
        return out

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._model = model
        cfg_path = self.config_path or getattr(args, 'head_mask_config_path', None)
        self._log('=' * 80)
        self._log('PathMaskDiagnosticCallback: on_train_begin')
        self._log(f'  config_path = {cfg_path!r}')
        if not cfg_path:
            self._log('  ABORT: no head_mask_config_path set, nothing to verify')
            return
        expected = self._expected_heads_from_config(cfg_path)
        attached = self._attached_heads_from_model(model)
        self._log('')
        self._log('  Per-path summary (expected from JSON / attached on model):')
        for kind in ('v2v', 't2v', 't2t'):
            exp_set = expected[kind]
            att_dict = attached[kind]
            n_att = sum((len(h) for h in att_dict.values()))
            n_layers_att = len(att_dict)
            self._log(f'    {kind}: expected={len(exp_set)} heads in {len({L for L, _ in exp_set})} layers   attached={n_att} heads in {n_layers_att} layers')
        problems: List[str] = []
        for kind in ('v2v', 't2v', 't2t'):
            exp_set = expected[kind]
            att_set = {(L, h) for L, hs in attached[kind].items() for h in hs}
            missing = exp_set - att_set
            extra = att_set - exp_set
            if missing:
                problems.append(f'{kind}: {len(missing)} expected heads NOT attached (sample: {sorted(missing)[:5]})')
            if extra:
                problems.append(f'{kind}: {len(extra)} heads attached but NOT in JSON (sample: {sorted(extra)[:5]})')
        if problems:
            self._log('')
            self._log('  WARNING: expected vs attached MISMATCH')
            for p in problems:
                self._log(f'    * {p}')
        else:
            total = sum((len(s) for s in expected.values()))
            if total > 0:
                self._log(f'  OK: all {total} expected (path, layer, head) entries are attached')
            else:
                self._log('  config has no entries (empty composite); nothing to attach')
        self._log('')
        self._log('  Sample-layer attached indices:')
        sample_layers = sorted({L for kind in ('v2v', 't2v', 't2t') for L in attached[kind]})
        sample_layers = sample_layers[:1] + sample_layers[len(sample_layers) // 2:len(sample_layers) // 2 + 1] + sample_layers[-1:]
        for L in sorted(set(sample_layers)):
            parts = []
            for kind in ('v2v', 't2v', 't2t'):
                hs = attached[kind].get(L, ())
                if hs:
                    parts.append(f"{kind}={list(hs)[:5]}{('...' if len(hs) > 5 else '')} (n={len(hs)})")
            if parts:
                self._log(f"    layer {L:>2}: {'  '.join(parts)}")
        rt = getattr(model, '_ov_region_runtime', None)
        if rt is None:
            self._log("  WARNING: model._ov_region_runtime is None -- update_region_runtime won't refresh per-sample image-token boundaries")
        else:
            self._log(f"  OK: model._ov_region_runtime exists (initial state: {(dict(rt) if rt else '<empty>')})")
        self._log('=' * 80)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self._steps_seen >= self.max_steps_to_log:
            return
        self._steps_seen += 1
        step = state.global_step
        model = model or self._model
        rt = getattr(model, '_ov_region_runtime', None)
        last_log = state.log_history[-1] if state.log_history else {}
        loss = last_log.get('loss')
        grad_norm = last_log.get('grad_norm')
        self._log('')
        self._log('-' * 80)
        self._log(f'step {step}  (path-mask diag, log {self._steps_seen}/{self.max_steps_to_log})')
        if rt is not None and rt:
            self._log(f"  runtime: has_image={rt.get('has_image')}  orig_seq_len={rt.get('orig_seq_len')}  n_img_placeholders={rt.get('n_img_placeholders')}  img_placeholder_pos={rt.get('img_placeholder_pos')}")
        else:
            self._log('  runtime: <empty> -- update_region_runtime not called yet (unexpected by step end!)')
        self._log(f'  trainer log: loss={loss}  grad_norm={grad_norm}')
        if isinstance(loss, (int, float)):
            if loss != loss or loss in (float('inf'), float('-inf')):
                self._log('  FAIL: loss is NaN / Inf -- mask is corrupting forward / backward')
            else:
                self._log('  OK: loss is finite')
