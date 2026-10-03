import importlib.util
import os
import sys
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LM = os.path.join(ROOT, 'LLaVA-NeXT', 'llava', 'model', 'language_model')

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
plugin = _load('amp_standalone', os.path.join(LM, 'attention_mask_plugin.py'))
flexmod = _load('rfa_standalone', os.path.join(LM, 'region_flex_attention.py'))

def _runtime_for(vis_s, vis_e, kv_seq_len):
    return {'has_image': True, 'orig_seq_len': kv_seq_len, 'n_img_placeholders': vis_e - vis_s, 'img_placeholder_pos': vis_s}

def _run_case(name, H, T, vis_s, vis_e, v2v, t2v, t2t):
    rt = _runtime_for(vis_s, vis_e, T)
    dtype = torch.float32
    dense = plugin.build_region_attention_mask(None, bsz=1, num_heads=H, q_len=T, kv_seq_len=T, past_len=0, dtype=dtype, device=torch.device('cpu'), v2v_heads=v2v, t2t_heads=t2t, runtime=rt, t2v_heads=t2v)
    if dense is None:
        print(f'[{name}] dense returned None (no mask) -- skipping')
        return True
    allow = flexmod.region_allow_matrix(num_heads=H, q_len=T, kv_seq_len=T, past_len=0, vis_s=vis_s, vis_e=vis_e, v2v_heads=v2v, t2v_heads=t2v, t2t_heads=t2t, device='cpu')
    dense_allow = dense[0] == 0.0
    ok = torch.equal(dense_allow, allow)
    n_blocked = int((~allow).sum())
    print(f"[{name}] H={H} T={T} vis=[{vis_s},{vis_e}) v2v={v2v} t2v={t2v} t2t={t2t} blocked_cells={n_blocked}  -> {('PASS' if ok else 'FAIL')}")
    if not ok:
        diff = dense_allow != allow
        print('   first mismatch at:', torch.nonzero(diff)[:3].tolist())
    return ok

def _ordered_to_dense(num_blocks, indices):
    dense = torch.zeros_like(indices, dtype=torch.bool)
    valid = torch.arange(indices.shape[-1]).view(*(1,) * (indices.ndim - 1), -1) < num_blocks.unsqueeze(-1)
    dense.scatter_(-1, indices.to(torch.long), valid)
    return dense

def _run_block_case(name, H, q_len, kv_len, past_len, vis_s, vis_e, v2v_heads, t2v_heads, t2t_heads):
    from torch.nn.attention.flex_attention import BlockMask

    def _head_set(idxs):
        result = torch.zeros(H, dtype=torch.bool)
        if idxs:
            result[torch.tensor(idxs, dtype=torch.long)] = True
        return result
    v2v = _head_set(v2v_heads)
    t2v = _head_set(t2v_heads)
    t2t = _head_set(t2t_heads)

    def unused_mask_mod(b, h, q_idx, kv_idx):
        return q_idx.new_ones((), dtype=torch.bool)
    block = flexmod._build_analytic_block_mask(BlockMask, unused_mask_mod, num_heads=H, q_len=q_len, kv_seq_len=kv_len, past_len=past_len, vis_s=vis_s, vis_e=vis_e, v2v=v2v, t2v=t2v, t2t=t2t, device=torch.device('cpu'))
    allow = flexmod.region_allow_matrix(num_heads=H, q_len=q_len, kv_seq_len=kv_len, past_len=past_len, vis_s=vis_s, vis_e=vis_e, v2v_heads=v2v_heads, t2v_heads=t2v_heads, t2t_heads=t2t_heads, device='cpu')
    block_size = block.BLOCK_SIZE[0]
    q_pad = -q_len % block_size
    kv_pad = -kv_len % block_size
    padded = torch.nn.functional.pad(allow, (0, kv_pad, 0, q_pad), value=False)
    token_counts = padded.view(H, padded.shape[-2] // block_size, block_size, padded.shape[-1] // block_size, block_size).sum(dim=(2, 4))
    expected_any = token_counts > 0
    expected_full = token_counts == block_size * block_size
    expected_partial = expected_any & ~expected_full
    actual_partial = _ordered_to_dense(block.kv_num_blocks, block.kv_indices)[0]
    actual_full = _ordered_to_dense(block.full_kv_num_blocks, block.full_kv_indices)[0]
    ok = torch.equal(actual_partial, expected_partial) and torch.equal(actual_full, expected_full)
    print(f"[{name}] analytic blocks q={q_len} kv={kv_len} past={past_len} partial={int(actual_partial.sum())} full={int(actual_full.sum())} -> {('PASS' if ok else 'FAIL')}")
    return ok

def main():
    res = []
    res.append(_run_case('v2v-only', 6, 12, 2, 8, [0, 1], None, None))
    res.append(_run_case('t2v-only', 6, 12, 2, 8, None, [2, 3], None))
    res.append(_run_case('t2t-only', 6, 12, 2, 8, None, None, [4]))
    res.append(_run_case('all-three-diff-heads', 6, 12, 2, 8, [0], [1, 2], [3, 4]))
    res.append(_run_case('same-head-multi-path', 6, 12, 2, 8, [0], [0], [0]))
    res.append(_run_case('v2v-all-heads', 4, 10, 1, 6, [0, 1, 2, 3], None, None))
    res.append(_run_case('bigger', 8, 40, 3, 27, [0, 1, 2], [3, 4], [5]))
    res.append(_run_block_case('analytic-prefill', 6, 310, 310, 0, 37, 255, [0, 1], [2, 3], [4]))
    res.append(_run_block_case('analytic-cached', 6, 55, 365, 310, 37, 255, [0, 1], [2, 3], [4]))
    print('-' * 60)
    if all(res):
        print(f'ALL {len(res)} CASES PASS: flex mask_mod == dense region mask')
        sys.exit(0)
    print(f'FAILED {res.count(False)}/{len(res)}')
    sys.exit(1)
if __name__ == '__main__':
    main()
