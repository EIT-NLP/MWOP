import json
import os
import time
from pathlib import Path
import torch
from triton.testing import _summarize_statistics
GUARD = None
LAST_SAMPLE = None
_RESIDENT = []

def set_resident_floor(gib):
    if os.environ.get('MWOP_RESERVE_VRAM') != '1':
        return
    chunk = 256 * 1024 ** 2
    count = int(gib * 1024 ** 3 // chunk)
    while len(_RESIDENT) > count:
        _RESIDENT.pop()
    while len(_RESIDENT) < count:
        _RESIDENT.append(torch.empty(chunk, dtype=torch.uint8, device='cuda'))

def check_guard():
    if GUARD is not None:
        GUARD()

def top_up_cache(tag='phase'):
    if os.environ.get('MWOP_RESERVE_VRAM') != '1':
        return
    margin = int(os.environ.get('MWOP_FREE_MIB', '512')) * 1024 ** 2
    buffers = []
    try:
        while True:
            free, total = torch.cuda.mem_get_info()
            size = min(128 * 1024 ** 2, free - margin)
            if size < 16 * 1024 ** 2:
                break
            buffers.append(torch.empty(size, dtype=torch.uint8, device='cuda'))
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError('VRAM reservation raced with an allocation')
    finally:
        buffers.clear()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    row = dict(epoch=time.time(), phase=os.environ.get('MWOP_PHASE'), tag=tag, free_bytes=free, total_bytes=total, occupied_fraction=1 - free / total, torch_allocated_bytes=torch.cuda.memory_allocated(), torch_reserved_bytes=torch.cuda.memory_reserved(), resident_floor_bytes=sum((x.numel() for x in _RESIDENT)))
    if os.environ.get('MWOP_MEMORY_LOG'):
        with Path(os.environ['MWOP_MEMORY_LOG']).open('a') as f:
            f.write(json.dumps(row) + '\n')
    assert free <= margin + 16 * 1024 ** 2, row

def do_bench_cudagraph(fn, rep=20, grad_to_none=None, quantiles=None, return_mode='mean'):
    global LAST_SAMPLE
    check_guard()
    assert return_mode in ['min', 'max', 'mean', 'median', 'all']
    with torch.cuda.stream(torch.cuda.Stream()):
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        if estimate_ms == 0:
            n_repeat = 1000
        else:
            n_repeat = max(1, int(rep / estimate_ms))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                fn()
        torch.cuda.synchronize()
        top_up_cache('before_timed_replay')
        ret = []
        n_retries = 10
        for _ in range(n_retries):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            ret += [start_event.elapsed_time(end_event) / n_repeat]
        check_guard()
        LAST_SAMPLE = dict(batch_mean_ms=list(ret), replay_count_per_batch=n_repeat, estimate_ms=estimate_ms, target_rep_ms=rep)
        return _summarize_statistics(ret, quantiles, return_mode)
