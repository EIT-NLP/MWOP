import torch
import triton
import triton.language as tl

@triton.jit
def _update(q, K, V, base, ki, qi, end, acc, z, m, HK: tl.constexpr, D: tl.constexpr, MASKED: tl.constexpr):
    di = tl.arange(0, D)
    ptr = base + ki[:, None] * HK * D + di[None, :]
    if MASKED:
        k = tl.load(K + ptr, mask=(ki < end)[:, None], other=0)
    else:
        k = tl.load(K + ptr)
    scores = tl.dot(q, tl.trans(k)) * (D ** (-0.5) * 1.4426950408889634)
    if MASKED:
        scores = tl.where((ki[None, :] < end) & (ki[None, :] <= qi[:, None]), scores, -float('inf'))
    nm = tl.maximum(m, tl.max(scores, 1))
    if MASKED:
        safe = tl.where(nm == -float('inf'), 0.0, nm)
    else:
        safe = nm
    p = tl.exp2(scores - safe[:, None])
    alpha = tl.exp2(m - safe)
    acc = acc * alpha[:, None]
    if MASKED:
        v = tl.load(V + ptr, mask=(ki < end)[:, None], other=0)
    else:
        v = tl.load(V + ptr)
    acc = tl.dot(p.to(q.dtype), v, acc)
    z = z * alpha + tl.sum(p, 1)
    return (acc, z, nm)

@triton.jit
def _ranges(Q, K, V, O, Flags, L: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, VS: tl.constexpr, VE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, OFFBAND: tl.constexpr, INTERLEAVE: tl.constexpr):
    np: tl.constexpr = triton.cdiv(VS, BM)
    nv: tl.constexpr = triton.cdiv(VE - VS, BM)
    nt: tl.constexpr = np + nv + triton.cdiv(L - VE, BM)
    if INTERLEAVE:
        tile, h = (tl.program_id(0) // HQ, tl.program_id(0) % HQ)
    else:
        tile, h = (tl.program_id(0) % nt, tl.program_id(0) // nt)
    b = tl.program_id(1)
    role = tl.where(tile < np, 0, tl.where(tile < np + nv, 1, 2))
    qs = tl.where(role == 0, tile * BM, tl.where(role == 1, VS + (tile - np) * BM, VE + (tile - np - nv) * BM))
    qe = tl.where(role == 0, VS, tl.where(role == 1, VE, L))
    qi = qs + tl.arange(0, BM)
    qptr = b * L * HQ * D + qi[:, None] * HQ * D + h * D + tl.arange(0, D)[None, :]
    q = tl.load(Q + qptr, mask=(qi < qe)[:, None], other=0)
    vv = tl.load(Flags + h)
    tv = tl.load(Flags + HQ + h)
    tt = tl.load(Flags + 2 * HQ + h)
    end = tl.where(role == 0, VS, tl.where(role == 1, tl.where(vv, VS, VE), tl.where(tv, VS, tl.where(tt, VE, L))))
    limit = tl.minimum(end, tl.minimum(qs + BM, qe))
    base = b * L * HK * D + h // (HQ // HK) * D
    m = tl.full((BM,), -float('inf'), tl.float32)
    z = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    if OFFBAND:
        full = tl.minimum(qs, end) // BN * BN
        for start in tl.range(0, full, BN):
            ki = start + tl.arange(0, BN)
            acc, z, m = _update(q, K, V, base, ki, qi, end, acc, z, m, HK, D, False)
    else:
        full = 0
    for start in tl.range(full, limit, BN):
        ki = start + tl.arange(0, BN)
        acc, z, m = _update(q, K, V, base, ki, qi, end, acc, z, m, HK, D, True)
    if (role == 2) & tv & ~tt:
        for start in tl.range(VE, tl.minimum(qs + BM, L), BN):
            ki = start + tl.arange(0, BN)
            acc, z, m = _update(q, K, V, base, ki, qi, L, acc, z, m, HK, D, True)
    out = acc / tl.where(z > 0.0, z, 1.0)[:, None]
    tl.store(O + qptr, out.to(O.dtype.element_ty), mask=(qi < qe)[:, None])

def _launch_attention(q, k, v, flags, span, config=(64, 64, 4, 2), offband=True, interleave=False):
    b, l, h, d = q.shape
    s, e = span
    bm, bn, w, st = config
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert k.shape == v.shape and k.shape[:2] == (b, l) and (d == 128)
    assert flags.shape == (3, h) and flags.is_contiguous() and (0 <= s <= e <= l)
    out = torch.empty_like(q)
    nt = triton.cdiv(s, bm) + triton.cdiv(e - s, bm) + triton.cdiv(l - e, bm)
    _ranges[nt * h, b](q, k, v, out, flags, l, h, k.shape[2], d, s, e, bm, bn, offband, interleave, num_warps=w, num_stages=st)
    return out

def attention(q, k, v, flags, span, config=(128, 32, 4, 2)):
    if isinstance(config, dict):
        kernel = config['kernel']
        tile = tuple(config['config'])
    else:
        kernel = 'range_interleave'
        tile = tuple(config)
    if tile[2] != 4:
        raise ValueError('MWOP attention uses exactly four warps')
    if kernel not in ('range_masked', 'range_offband', 'range_interleave'):
        raise ValueError(kernel)
    return _launch_attention(q, k, v, flags, span, tile, offband=kernel != 'range_masked', interleave=kernel == 'range_interleave')
