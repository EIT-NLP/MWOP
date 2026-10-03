from __future__ import annotations
import argparse
import csv
import os
import random
import sys
import time
import numpy as np
import torch
PATHS = ('v2v', 't2v', 't2t')

class TaylorAccumulator:

    def __init__(self, num_layers, num_heads):
        self.num_layers, self.num_heads = (num_layers, num_heads)
        shape = (num_layers, num_heads, 3)
        self.count = np.zeros(shape, dtype=np.int64)
        self.abs_sum = np.zeros(shape, dtype=np.float64)
        self.abs_sum_pq = np.zeros(shape, dtype=np.float64)

    def update_batch(self, li, taylor_per_head, q_counts):
        finite = np.isfinite(taylor_per_head)
        self.abs_sum[li] += np.abs(np.where(finite, taylor_per_head, 0.0))
        self.count[li] += finite.astype(np.int64)
        for pi, qc in enumerate(q_counts):
            if qc > 0:
                pq_finite = finite[:, pi]
                self.abs_sum_pq[li, :, pi] += np.abs(np.where(pq_finite, taylor_per_head[:, pi] / qc, 0.0))

    def save_csv(self, path):
        n_safe = np.where(self.count > 0, self.count, 1)
        abs_mean = self.abs_sum / n_safe
        abs_mean_pq = self.abs_sum_pq / n_safe
        mask = self.count == 0
        abs_mean[mask] = np.nan
        abs_mean_pq[mask] = np.nan
        cols = ['layer', 'head']
        for p in PATHS:
            cols += [f'taylor_{p}_abs_mean', f'taylor_{p}_abs_mean_pq', f'taylor_{p}_count']
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(cols)
            for li in range(self.num_layers):
                for hi in range(self.num_heads):
                    row = [li, hi]
                    for pi, _ in enumerate(PATHS):
                        row += [float(abs_mean[li, hi, pi]), float(abs_mean_pq[li, hi, pi]), int(self.count[li, hi, pi])]
                    w.writerow(row)

def block_taylor_vec(attn, v_per_head, g_per_head, q_mask, k_mask):
    q_count = int(q_mask.sum().item())
    if q_count == 0 or int(k_mask.sum().item()) == 0:
        return (torch.zeros(attn.shape[0], device=attn.device), 0)
    a_block = attn[:, q_mask][:, :, k_mask].float()
    v_block = v_per_head[:, k_mask].float()
    z_block = torch.bmm(a_block, v_block)
    g_block = g_per_head[:, q_mask].float()
    return ((g_block * z_block).sum(dim=(-2, -1)), q_count)

def load_task_samples(task, n_samples, seed):
    from lmms_eval.tasks import get_task_dict, TaskManager
    tm = TaskManager()
    td = get_task_dict([task], tm)
    obj = td[task]
    while isinstance(obj, (tuple, list)):
        obj = obj[-1]
    if isinstance(obj, dict):
        obj = next(iter(obj.values()))
    docs = obj.test_docs() if getattr(obj, 'has_test_docs', lambda: False)() else obj.validation_docs()
    indices = list(range(len(docs)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    out = []
    for index in indices:
        if len(out) >= n_samples:
            break
        try:
            d = docs[index]
            vis = obj.doc_to_visual(d)
            img = vis[0] if isinstance(vis, (list, tuple)) and vis else vis
            q = obj.doc_to_text(d)
            a = obj.doc_to_target(d)
            if isinstance(q, (list, tuple)):
                q = q[0]
            if isinstance(a, (list, tuple)):
                a = a[0] if a else ''
            q, a = (str(q).strip(), str(a).strip())
            if img is not None and q and a:
                out.append({'image': img, 'question': q, 'answer': a})
        except Exception as e:
            print(f'[collect] doc skipped: {type(e).__name__}: {e}', flush=True)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pretrained', default=None, help='default = base OV ckpt under checkpoints/')
    ap.add_argument('--task', required=True, help='lmms-eval task name (gqa, textvqa_val, ...)')
    ap.add_argument('--n_samples', type=int, default=128)
    ap.add_argument('--conv_template', default='qwen_1_5')
    ap.add_argument('--device', default='auto', help="'auto' shards across all GPUs; or 'cuda:0' for one")
    ap.add_argument('--max_image_side', type=int, default=768, help="cap the longest image side (px); bounds anyres patches/seq-len so output_attentions doesn't OOM on high-res samples. 0 = no cap.")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    project_root = os.environ.get('PROJECT_ROOT', os.getcwd())
    for p in (os.path.join(project_root, '..'),):
        if p not in sys.path:
            sys.path.insert(0, p)
    pretrained = args.pretrained or os.path.join(project_root, 'checkpoints/ckpts-lmm/llava-onevision-qwen2-7b-ov')
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
    from llava.conversation import conv_templates
    print('[collect] VERSION=diag5 (per-sample graph cleanup -> no OOM accumulation)', flush=True)
    print(f'[collect] task={args.task} n={args.n_samples} model={pretrained}', flush=True)
    dmap = 'auto' if args.device in ('auto', '', None) else args.device
    tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, get_model_name_from_path(pretrained), device_map=dmap, attn_implementation='eager', multimodal=True, torch_dtype='bfloat16')
    device = model.get_input_embeddings().weight.device
    for p in model.parameters():
        p.requires_grad_(False)

    def _enable_grad_on_emb(_m, _i, o):
        if isinstance(o, torch.Tensor):
            o.requires_grad_(True)
        return o
    model.get_input_embeddings().register_forward_hook(_enable_grad_on_emb)
    num_layers = len(model.model.layers)
    sa = model.model.layers[0].self_attn
    num_heads = sa.num_heads
    num_kv_heads = getattr(sa, 'num_key_value_heads', num_heads)
    head_dim = sa.head_dim
    kv_groups = num_heads // max(num_kv_heads, 1)
    print(f'[collect] layers={num_layers} heads={num_heads} kv={num_kv_heads} d={head_dim} groups={kv_groups}', flush=True)
    v_out, o_grad = ({}, {})
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.v_proj.register_forward_hook((lambda li: lambda mod, inp, out: v_out.__setitem__(li, (out[0] if isinstance(out, tuple) else out).detach()))(li))
        layer.self_attn.o_proj.register_full_backward_hook((lambda li: lambda mod, gi, go: o_grad.__setitem__(li, gi[0].detach()) if gi and gi[0] is not None else None)(li))
    samples = load_task_samples(args.task, args.n_samples, args.seed)
    print(f'[collect] loaded {len(samples)} samples for {args.task}', flush=True)
    if not samples:
        raise RuntimeError(f'no usable samples for task {args.task}')
    acc = TaylorAccumulator(num_layers, num_heads)
    conv_templ = conv_templates[args.conv_template]
    processed = skipped = 0
    t0 = time.time()
    for idx, s in enumerate(samples):
        image, question, answer = (s['image'], s['question'], s['answer'])
        if args.max_image_side > 0 and hasattr(image, 'thumbnail') and hasattr(image, 'size') and (max(image.size) > args.max_image_side):
            image = image.copy()
            image.thumbnail((args.max_image_side, args.max_image_side))
        cq = conv_templ.copy()
        cq.append_message(cq.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + question)
        cq.append_message(cq.roles[1], None)
        cqa = conv_templ.copy()
        cqa.append_message(cqa.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + question)
        cqa.append_message(cqa.roles[1], answer)
        prompt_q, prompt_qa = (cq.get_prompt(), cqa.get_prompt())
        try:
            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [t.to(dtype=torch.bfloat16, device=device) for t in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.bfloat16, device=device)
            ids_q = tokenizer_image_token(prompt_q, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            ids_qa = tokenizer_image_token(prompt_qa, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            labels = ids_qa.clone()
            labels[0, :ids_q.shape[1]] = IGNORE_INDEX
            img_mask = ids_qa[0] == IMAGE_TOKEN_INDEX
            n_img_ph = int(img_mask.sum().item())
            img_ph_pos = int(img_mask.nonzero(as_tuple=True)[0][0].item()) if n_img_ph >= 1 else 0
            v_out.clear()
            o_grad.clear()
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids=ids_qa, labels=labels, images=image_tensor, image_sizes=[[image.size[0], image.size[1]]], output_attentions=True, use_cache=False)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                skipped += 1
                if skipped <= 3:
                    print(f'[collect] skip idx={idx} reason=loss  loss={loss}', flush=True)
                continue
            loss.backward()
            attentions = outputs.attentions
            if attentions is None or attentions[0] is None:
                skipped += 1
                if skipped <= 3:
                    print(f'[collect] skip idx={idx} reason=attentions_none  (has_attn={attentions is not None})', flush=True)
                continue
            T_final = attentions[0].shape[-1]
            n_img = T_final - ids_qa.shape[1] + n_img_ph
            vis_s, vis_e = (img_ph_pos, img_ph_pos + max(n_img, 0))
            if vis_s >= vis_e:
                skipped += 1
                if skipped <= 3:
                    print(f'[collect] skip idx={idx} reason=vis_range  T_final={T_final} ids_qa_len={ids_qa.shape[1]} n_img_ph={n_img_ph} img_ph_pos={img_ph_pos} n_img={n_img} vis_s={vis_s} vis_e={vis_e}', flush=True)
                continue
            vis_mask = torch.zeros(T_final, dtype=torch.bool, device=device)
            vis_mask[vis_s:vis_e] = True
            txt_mask = torch.zeros(T_final, dtype=torch.bool, device=device)
            txt_mask[vis_e:] = True
            with torch.no_grad():
                for li in range(num_layers):
                    attn, v_t, g_t = (attentions[li], v_out.get(li), o_grad.get(li))
                    if attn is None or v_t is None or g_t is None:
                        continue
                    ah = attn[0]
                    vm = vis_mask.to(ah.device)
                    tm = txt_mask.to(ah.device)
                    v_t, g_t = (v_t.to(ah.device), g_t.to(ah.device))
                    v_ph = v_t.view(v_t.shape[0], v_t.shape[1], num_kv_heads, head_dim).permute(0, 2, 1, 3)[0]
                    if kv_groups > 1:
                        v_ph = v_ph.repeat_interleave(kv_groups, dim=0)
                    g_ph = g_t.view(g_t.shape[0], g_t.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)[0]
                    iv, qv = block_taylor_vec(ah, v_ph, g_ph, vm, vm)
                    it, qtv = block_taylor_vec(ah, v_ph, g_ph, tm, vm)
                    ig, qtt = block_taylor_vec(ah, v_ph, g_ph, tm, tm)
                    acc.update_batch(li, torch.stack([iv, it, ig], dim=-1).cpu().numpy(), (qv, qtv, qtt))
            processed += 1
        except torch.cuda.OutOfMemoryError:
            skipped += 1
            if skipped <= 3:
                print(f'[collect] skip idx={idx} reason=OOM', flush=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f'[collect] skip idx={idx}: {type(e).__name__}: {e}', flush=True)
            skipped += 1
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        finally:
            v_out.clear()
            o_grad.clear()
            outputs = None
            loss = None
            attentions = None
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        if (idx + 1) % 16 == 0:
            print(f'[collect] {idx + 1}/{len(samples)} processed={processed} skipped={skipped} ({(time.time() - t0) / (idx + 1):.2f}s/sample)', flush=True)
    dt = time.time() - t0
    acc.save_csv(args.out)
    print(f'[collect] DONE task={args.task} processed={processed} skipped={skipped} wall={dt:.1f}s ({dt / max(processed, 1):.2f}s/processed) -> {args.out}', flush=True)
if __name__ == '__main__':
    main()
