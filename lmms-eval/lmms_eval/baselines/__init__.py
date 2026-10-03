import os
from lmms_eval.baselines.loader import load_baseline
from lmms_eval.baselines.registry import BASELINE_REGISTRY

def get_baseline_display_name(baseline_arg: str) -> str:
    if ':' in baseline_arg and (not baseline_arg.startswith('hf://')):
        model_name, task = baseline_arg.split(':', 1)
        if model_name in BASELINE_REGISTRY:
            return model_name
    if baseline_arg in BASELINE_REGISTRY:
        return baseline_arg
    if baseline_arg.startswith('hf://'):
        parts = baseline_arg[5:].split('/')
        return '/'.join(parts[:2]) if len(parts) >= 2 else baseline_arg
    if '/' in baseline_arg or '\\' in baseline_arg:
        filename = os.path.basename(baseline_arg)
        return os.path.splitext(filename)[0][:30]
    return baseline_arg
__all__ = ['BASELINE_REGISTRY', 'get_baseline_display_name', 'load_baseline']
