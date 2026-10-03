from __future__ import annotations
import argparse
import os
import signal
import sys
import torch
from torch.utils.data import DataLoader, Subset

def _add_paths(project_root: str):
    for p in (os.path.join(project_root, '..'), os.path.join(project_root, 'scripts', 'prune')):
        if p not in sys.path:
            sys.path.insert(0, p)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='merged HF model dir')
    ap.add_argument('--model_name', default='llava_qwen', help="must contain 'qwen' so the builder picks LlavaQwen")
    ap.add_argument('--data_path', required=True, help='training yaml or jsonl (calibration source)')
    ap.add_argument('--image_folder', required=True)
    ap.add_argument('--conv_version', default='qwen_1_5', help='conversation template key (matches training --version)')
    ap.add_argument('--num_samples', type=int, default=512, help='calibration samples to forward (bs=1, forward only)')
    ap.add_argument('--head_mask_config', default='', help="optional head ablation (zero-mask) applied BEFORE collection so a head-pruned/recovered model's FFN activations are measured in its real regime")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_tokens', type=int, default=0, help='if >0, cap tokens used per sample (truncate) to bound memory')
    ap.add_argument('--mode', default='activation', choices=['activation', 'taylor'], help='activation: mean|h_i| (forward-only, cheap). taylor: mean|h_i * dL/dh_i| (forward+backward, first-order LLM-Pruner importance = how much zeroing the neuron perturbs the loss).')
    ap.add_argument('--split_modality', action='store_true', help="ALSO bucket per-neuron importance by TOKEN MODALITY (vision vs text) using the expanded image-token block position -> adds scores_vision / scores_text to the output (the FFN analog of V2V/T2V/T2T, to test whether FFN importance is modality-specific). 'scores' (the all-token total used by the sweep) is byte-identical to a non-split run.")
    ap.add_argument('--ckpt_every', type=int, default=25, help='RESUMABLE: save a <out>.ckpt (raw accumulators + next index) every this many samples, and on SIGTERM/SIGINT. Re-running auto-resumes from it -- survives a time-limited interactive session. 0 disables.')
    ap.add_argument('--no_resume', action='store_true', help='ignore any existing <out>.ckpt and start a fresh collection.')
    args = ap.parse_args()
    project_root = os.environ.get('PROJECT_ROOT', os.getcwd())
    _add_paths(project_root)
    from llava.model.builder import load_pretrained_model
    from llava import conversation as conversation_lib
    from llava.train.train import DataArguments, LazySupervisedDataset, DataCollatorForSupervisedDataset
    try:
        from llava.constants import IMAGE_TOKEN_INDEX
    except Exception:
        IMAGE_TOKEN_INDEX = -200
    print(f'[collect] loading model: {args.ckpt}', flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.ckpt, None, args.model_name, device_map='auto', attn_implementation='sdpa')
    model.eval()
    dev = next(model.parameters()).device
    mdtype = next(model.parameters()).dtype
    cfg = model.config
    if args.head_mask_config:
        from llava.model.language_model.ablation_registry import apply_ablation_config
        summ = apply_ablation_config(model, args.head_mask_config)
        print(f'[collect] applied head mask: {args.head_mask_config} -> {summ}', flush=True)
    if args.conv_version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_version]
    else:
        print(f"[collect] WARN: conv_version '{args.conv_version}' not found; using current default '{conversation_lib.default_conversation.version}'", flush=True)
    da = DataArguments(data_path=args.data_path, image_folder=args.image_folder)
    da.image_processor = image_processor
    da.is_multimodal = True
    da.lazy_preprocess = True
    da.mm_use_im_start_end = getattr(cfg, 'mm_use_im_start_end', False)
    da.image_aspect_ratio = getattr(cfg, 'image_aspect_ratio', 'square')
    da.image_grid_pinpoints = getattr(cfg, 'image_grid_pinpoints', None)
    da.image_crop_resolution = getattr(cfg, 'image_crop_resolution', None)
    da.image_split_resolution = getattr(cfg, 'image_split_resolution', None)
    ds = LazySupervisedDataset(tokenizer=tokenizer, data_path=args.data_path, data_args=da)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    n = min(args.num_samples, len(ds))
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    ckpt_path = args.out + '.ckpt'
    fingerprint = f'{os.path.abspath(args.ckpt)}|n={n}|seed={args.seed}|mode={args.mode}|split={int(args.split_modality)}|hm={args.head_mask_config}|maxtok={args.max_tokens}'
    resume_obj, start_i, seen0 = (None, 0, 0)
    if not args.no_resume and os.path.isfile(ckpt_path):
        try:
            ro = torch.load(ckpt_path, map_location='cpu')
            if ro.get('fingerprint') == fingerprint:
                resume_obj = ro
                start_i = int(ro.get('next_i', 0))
                seen0 = int(ro.get('seen', 0))
                print(f'[collect] RESUME {ckpt_path}: {start_i}/{n} iterated, seen={seen0}', flush=True)
            else:
                print('[collect] checkpoint fingerprint mismatch -> ignoring (fresh start)', flush=True)
        except Exception as e:
            print(f'[collect] WARN unreadable checkpoint ({e}); fresh start', flush=True)
    idx_todo = idx[start_i:]
    loader = DataLoader(Subset(ds, idx_todo), batch_size=1, shuffle=False, collate_fn=collator, num_workers=4)
    print(f'[collect] dataset={len(ds)} -> {n} calibration samples ({len(idx_todo)} remaining)', flush=True)
    import re as _re
    _DOWN_RE = _re.compile('(?:^|\\.)layers\\.(\\d+)\\.mlp\\.down_proj$')
    taylor = args.mode == 'taylor'
    acc, cnt, hooks = ({}, {}, [])
    accv, cntv, acct, cntt = ({}, {}, {}, {})
    fwd_cache = {}
    split = args.split_modality
    cur = {}
    split_tally = {'img': 0, 'text': 0, 'multi': 0}
    if resume_obj is not None:

        def _restore(dst_acc, dst_cnt, akey, ckey):
            for _li, _t in resume_obj.get(akey, {}).items():
                dst_acc[int(_li)] = _t.clone()
            dst_cnt.update({int(_k): int(_v) for _k, _v in resume_obj.get(ckey, {}).items()})
        _restore(acc, cnt, 'acc', 'cnt')
        if split:
            _restore(accv, cntv, 'accv', 'cntv')
            _restore(acct, cntt, 'acct', 'cntt')
        for _k, _v in resume_obj.get('split_tally', {}).items():
            split_tally[_k] = int(_v)

    def _accum(d_acc, d_cnt, li, s, ntok):
        if li not in d_acc:
            d_acc[li] = torch.zeros_like(s)
            d_cnt[li] = 0
        elif d_acc[li].device != s.device:
            d_acc[li] = d_acc[li].to(s.device)
        d_acc[li] += s
        d_cnt[li] += ntok

    def _vrange(T):
        p, L, nph = (cur.get('p', -1), cur.get('L', 0), cur.get('nph', 0))
        if nph == 0:
            return (0, 0)
        if nph != 1 or p < 0:
            return None
        n_img = T - L + 1
        if not (0 <= p < T and 0 < n_img and (p + n_img <= T)):
            return None
        return (p, p + n_img)

    def accumulate_tokens(li, contrib):
        total = contrib.sum(dim=0)
        _accum(acc, cnt, li, total, contrib.shape[0])
        if not split:
            return
        T = contrib.shape[0]
        if cur.get('vr_T') != T:
            cur['vr'] = _vrange(T)
            cur['vr_T'] = T
        vr = cur['vr']
        if vr is None:
            return
        vs, ve = vr
        nvis = ve - vs
        if nvis > 0:
            vsum = contrib[vs:ve].sum(dim=0)
            _accum(accv, cntv, li, vsum, nvis)
            _accum(acct, cntt, li, total - vsum, T - nvis)
        else:
            _accum(acct, cntt, li, total, T)

    def make_pre_hook(li):

        def pre_hook(_mod, inp):
            h = inp[0]
            if taylor:
                fwd_cache[li] = h.detach().float()
            else:
                accumulate_tokens(li, h.detach().float().reshape(-1, h.shape[-1]).abs())
        return pre_hook

    def make_bwd_hook(li):

        def bwd_hook(_mod, grad_input, _grad_output):
            h = fwd_cache.pop(li, None)
            g = None
            for gi in grad_input:
                if gi is not None and gi.shape[-1] == (h.shape[-1] if h is not None else -1):
                    g = gi
                    break
            if h is None or g is None:
                return
            accumulate_tokens(li, (h * g.detach().float()).reshape(-1, h.shape[-1]).abs())
        return bwd_hook
    for name, mod in model.named_modules():
        m = _DOWN_RE.search(name)
        if m is not None:
            li = int(m.group(1))
            hooks.append(mod.register_forward_pre_hook(make_pre_hook(li)))
            if taylor:
                hooks.append(mod.register_full_backward_hook(make_bwd_hook(li)))
    if not hooks:
        raise RuntimeError("found no '*.layers.N.mlp.down_proj' modules to hook")
    print(f"[collect] mode={args.mode} hooked {len(acc) or 28} layers' down_proj", flush=True)
    if taylor:
        model.get_input_embeddings().weight.requires_grad_(True)
    DROP = {'prompts', 'id', 'ids'}
    KEEP = {'input_ids', 'attention_mask', 'labels', 'images', 'image_sizes', 'modalities'}
    seen = seen0

    def _save_ckpt(next_i):
        if args.ckpt_every <= 0:
            return
        obj = {'version': 1, 'fingerprint': fingerprint, 'mode': args.mode, 'split': split, 'next_i': int(next_i), 'seen': int(seen), 'n': n, 'split_tally': dict(split_tally), 'acc': {li: t.detach().to('cpu', torch.float32) for li, t in acc.items()}, 'cnt': dict(cnt)}
        if split:
            obj['accv'] = {li: t.detach().to('cpu', torch.float32) for li, t in accv.items()}
            obj['cntv'] = dict(cntv)
            obj['acct'] = {li: t.detach().to('cpu', torch.float32) for li, t in acct.items()}
            obj['cntt'] = dict(cntt)
        tmp = ckpt_path + '.tmp'
        torch.save(obj, tmp)
        os.replace(tmp, ckpt_path)
    _stop = {'flag': False}

    def _sig(signum, _frame):
        _stop['flag'] = True
        print(f'[collect] signal {signum} -> checkpoint & stop after current sample', flush=True)
    for _s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_s, _sig)
        except Exception:
            pass
    grad_ctx = torch.enable_grad() if taylor else torch.no_grad()
    with grad_ctx:
        for i, batch in enumerate(loader):
            global_i = start_i + i
            out_obj = loss = None
            try:
                if split:
                    _ids = batch['input_ids'][0]
                    _ip = (_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                    cur.clear()
                    cur['p'] = int(_ip[0]) if len(_ip) else -1
                    cur['L'] = int(_ids.shape[0])
                    cur['nph'] = int(len(_ip))
                    split_tally['img' if cur['nph'] == 1 else 'text' if cur['nph'] == 0 else 'multi'] += 1
                fwd = {}
                for k, v in batch.items():
                    if k in DROP or k not in KEEP:
                        continue
                    if k == 'images':
                        fwd[k] = [im.to(device=dev, dtype=mdtype) if torch.is_tensor(im) else im for im in v]
                    elif torch.is_tensor(v):
                        t = v.to(dev)
                        if args.max_tokens > 0 and k in ('input_ids', 'attention_mask', 'labels') and (t.dim() == 2):
                            t = t[:, :args.max_tokens]
                        fwd[k] = t
                    else:
                        fwd[k] = v
                out_obj = model(use_cache=False, **fwd)
                if taylor:
                    loss = out_obj.loss if hasattr(out_obj, 'loss') else out_obj['loss']
                    if loss is not None and loss.requires_grad:
                        loss.backward()
                seen += 1
            except Exception as e:
                print(f'[collect] WARN sample {global_i} skipped: {type(e).__name__}: {e}', flush=True)
            finally:
                out_obj = loss = None
                fwd_cache.clear()
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            done_i = global_i + 1
            if done_i % 50 == 0:
                print(f'[collect] {done_i}/{n} iterated (ok={seen})', flush=True)
            if args.ckpt_every > 0 and done_i % args.ckpt_every == 0:
                _save_ckpt(done_i)
            if _stop['flag']:
                _save_ckpt(done_i)
                print(f'[collect] EARLY STOP at {done_i}/{n}; checkpoint saved -> re-run to resume', flush=True)
                break
    for h in hooks:
        h.remove()
    if _stop['flag']:
        print(f'[collect] interrupted -> partial backup at {ckpt_path}; rerun the same command to resume')
        return
    if not acc:
        raise RuntimeError('no scores captured -- did any forward/backward succeed?')
    _save_ckpt(n)
    inter = int(next(iter(acc.values())).numel())
    scores = {li: (acc[li] / max(cnt[li], 1)).cpu() for li in sorted(acc)}
    metric_label = 'taylor' if taylor else 'absmean'
    layer_means = {li: float(scores[li].mean()) for li in scores}
    out = {'metric': metric_label, 'mode': args.mode, 'num_samples': seen, 'intermediate_size': inter, 'layers': len(scores), 'ckpt': args.ckpt, 'head_mask_config': args.head_mask_config, 'scores': scores}
    if split:
        out['scores_vision'] = {li: (accv[li] / max(cntv.get(li, 0), 1)).cpu() for li in sorted(accv)}
        out['scores_text'] = {li: (acct[li] / max(cntt.get(li, 0), 1)).cpu() for li in sorted(acct)}
        out['n_vision_tokens'] = sum(cntv.values()) // max(len(cntv), 1)
        out['n_text_tokens'] = sum(cntt.values()) // max(len(cntt), 1)
        out['split_tally'] = split_tally
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f'[collect] DONE mode={args.mode} ok={seen}/{n} layers={len(scores)} inter={inter}', flush=True)
    lm = sorted(layer_means.items())
    _lbl = 'mean|h*dL/dh|' if taylor else 'mean|h|'
    print(f'[collect] per-layer {_lbl} (first 4): ' + ', '.join((f'L{li}={v:.4g}' for li, v in lm[:4])))
    print(f'[collect] per-layer {_lbl} (last 4):  ' + ', '.join((f'L{li}={v:.4g}' for li, v in lm[-4:])))
    if split:
        print(f"[collect] split_modality: samples img={split_tally['img']} text={split_tally['text']} multi(skipped)={split_tally['multi']}  vis_tok/layer={out['n_vision_tokens']} txt_tok/layer={out['n_text_tokens']}", flush=True)
        if out['scores_vision'] and out['scores_text']:
            vmean = sum((float(t.mean()) for t in out['scores_vision'].values())) / len(out['scores_vision'])
            tmean = sum((float(t.mean()) for t in out['scores_text'].values())) / len(out['scores_text'])
            print(f'[collect] mean vision-importance={vmean:.4g}  mean text-importance={tmean:.4g}')
    print(f'[collect] wrote {args.out}')
if __name__ == '__main__':
    main()
