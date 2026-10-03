from PIL import ImageDraw
from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
COCO_METRICS = ['Bleu_4', 'Bleu_3', 'Bleu_2', 'Bleu_1', 'METEOR', 'ROUGE_L', 'CIDEr']
from loguru import logger as eval_logger

def refcoco_bbox_doc_to_visual(doc):
    bbox = doc['bbox']
    image = doc['image'].convert('RGB')
    draw = ImageDraw.Draw(image)
    bbox_xy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    draw.rectangle(bbox_xy, outline='red')
    return [image.convert('RGB')]

def refcoco_seg_doc_to_visual(doc):
    seg = doc['segmentation']
    image = doc['image'].convert('RGB')
    draw = ImageDraw.Draw(image)
    draw.polygon(seg)
    return [image.convert('RGB')]

def refcoco_doc_to_text(doc):
    return 'Provide a short description for this region.'

def refcoco_process_result(doc, result):
    pred = result[0] if len(result) > 0 else ''
    ann_id = doc['question_id']
    data_dict = {'answer': doc['answer'], 'pred': pred, 'ann_id': ann_id}
    return {f'refcoco_{metric}': data_dict for metric in COCO_METRICS}

def refcoco_aggregation_result(results, metric):
    scorers = [(Bleu(4), 'Bleu_1'), (Bleu(4), 'Bleu_2'), (Bleu(4), 'Bleu_3'), (Bleu(4), 'Bleu_4'), (Meteor(), 'METEOR'), (Rouge(), 'ROUGE_L'), (Cider(), 'CIDEr')]
    scorers_dict = {s[1]: s for s in scorers}
    stored_results = []
    dataset = {'annotations': [], 'images': []}
    idx = 0
    ann_id = 0
    for result in results:
        stored_results.append({'image_id': idx, 'caption': result['pred']})
        for s in result['answer']:
            dataset['annotations'].append({'image_id': idx, 'caption': s, 'id': ann_id})
            ann_id += 1
        dataset['images'].append({'id': idx})
        idx += 1
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    coco_result = coco.loadRes(stored_results)
    coco_eval = COCOEvalCap(coco, coco_result)
    imgIds = coco_eval.params['image_id']
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
        res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]
    eval_logger.info('tokenization...')
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    eval_logger.info(f'Computing {metric} scores...')
    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    if type(score) == list:
        n = int(metric.split('_')[-1])
        score = score[n - 1]
    return score

def refcoco_bleu4(results):
    return refcoco_aggregation_result(results, 'Bleu_4')

def refcoco_bleu3(results):
    return refcoco_aggregation_result(results, 'Bleu_3')

def refcoco_bleu2(results):
    return refcoco_aggregation_result(results, 'Bleu_2')

def refcoco_bleu1(results):
    return refcoco_aggregation_result(results, 'Bleu_1')

def refcoco_meteor(results):
    return refcoco_aggregation_result(results, 'METEOR')

def refcoco_rougel(results):
    return refcoco_aggregation_result(results, 'ROUGE_L')

def refcoco_cider(results):
    return refcoco_aggregation_result(results, 'CIDEr')

def refcoco_spice(results):
    return refcoco_aggregation_result(results, 'SPICE')
