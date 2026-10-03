import re

def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get('pre_prompt', '')
    post_prompt = lmms_eval_specific_kwargs.get('post_prompt', '\nPlease answer the question with a short answer.')
    question = doc['question']
    full_prompt = f'{pre_prompt}{question}{post_prompt}'
    return full_prompt

def doc_to_target(doc):
    answer = doc['answer']
    options = doc['options']
    answer_idx = ord(answer.upper()) - ord('A')
    if 0 <= answer_idx < len(options):
        opt = options[answer_idx]
        if '. ' in opt:
            return opt.split('. ', 1)[1].strip()
        return opt.strip()
    return answer

def _normalize_text(text):
    text = text.lower().strip()
    text = re.sub('[^\\w\\s]', '', text)
    text = re.sub('\\s+', ' ', text)
    return text

def process_results(doc, results):
    if not results:
        return {'acc_score': 0.0}
    target = doc_to_target(doc)
    target_normalized = _normalize_text(target)
    acc_score = 0.0
    for pred in results:
        pred_normalized = _normalize_text(pred)
        if target_normalized in pred_normalized or pred_normalized in target_normalized:
            acc_score += 1.0
    return {'acc_score': acc_score / len(results)}
