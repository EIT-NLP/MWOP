def chartqa_doc_to_visual(doc):
    return [doc['image'].convert('RGB')]

def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc['question']
    pre_prompt = lmms_eval_specific_kwargs['pre_prompt']
    post_prompt = lmms_eval_specific_kwargs['post_prompt']
    return f'{pre_prompt}{question}{post_prompt}'

def chartqa_process_results(doc, results):
    pred = results[0]
    type = doc['type']
    score = relaxed_correctness(pred, doc['answer'])
    score = 1.0 if score else 0.0
    return_dict = {'relaxed_overall': score}
    if type == 'human_test':
        return_dict['relaxed_human_split'] = score
    else:
        return_dict['relaxed_augmented_split'] = score
    return return_dict

def relaxed_correctness(prediction, target, max_relative_change: float=0.05) -> bool:

    def _to_float(text: str):
        try:
            if text.endswith('%'):
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()
