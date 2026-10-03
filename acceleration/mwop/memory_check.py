import argparse, json
from pathlib import Path
import torch
from .config import load_plans, flags_tensor
from .triton_prefill import attention
from .timing import top_up_cache
from .token_schedule import layout, override_plans

@torch.inference_mode()
def run(a):
    plans, _ = load_plans(a.config_dir)
    plans = override_plans(plans, a.scenario)
    selected = json.loads(Path(a.tiles).read_text())
    torch.manual_seed(37)
    checks = []
    shapes = layout(a.scenario)
    cases = list(dict.fromkeys(((r['length'], tuple(r['span'])) for r in shapes)))
    for length, span in cases + [(107, (18, 83)), (129, (0, 65)), (67, (0, 67)), (73, (18, 18)), (97, (0, 0))]:
        q = torch.randn(1, length, 28, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, length, 4, 128, device='cuda', dtype=q.dtype)
        v = torch.randn_like(k)
        top_up_cache('memory_check_shape')
        for layer in range(28):
            flags = flags_tensor(plans[layer], 'cuda')
            y = attention(q, k, v, flags, span, selected[str(layer)])
            assert torch.isfinite(y).all()
        torch.cuda.synchronize()
        checks.append(dict(length=length, span=span, layers=28))
        print('MEMORY_CHECK_SHAPE_COMPLETE', length, flush=True)
    Path(a.output).write_text(json.dumps(dict(status='complete', checks=checks), indent=2) + '\n')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', default='base')
    p.add_argument('--config-dir', default='../../pruning_config')
    p.add_argument('--tiles', required=True)
    p.add_argument('--output', required=True)
    run(p.parse_args())
