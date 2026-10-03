import re
from typing import Any, Dict, List, Optional
from lmms_eval.tasks._task_utils.default_template_yaml import load_default_template_yaml
config = load_default_template_yaml(__file__)

def _extract_answer_letter(text: str) -> str:
    text = text.strip()
    match = re.match('[\\(\\s]*([A-Z])[\\)\\.\\s]*', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ''

def blink_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]]=None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    num_choices = len(doc['choices'])
    choice_letters = ', '.join([chr(65 + i) for i in range(num_choices)])
    prompt = lmms_eval_specific_kwargs.get('pre_prompt', '').format(choice_letters) + doc['prompt']
    return prompt

def blink_doc_to_visual(doc: dict) -> list:
    keys = doc.keys()
    image_keys = [item for item in keys if re.match('^image_\\d+$', item)]
    image_list = []
    for image_key in image_keys:
        image = doc[image_key]
        if image is not None:
            image_list.append(image.convert('RGB'))
    return image_list

def blink_process_results(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    key_name = 'blink_acc'
    grounded_output = doc['answer'].strip('()')
    response = result[0]
    pred_letter = _extract_answer_letter(response)
    flag = pred_letter == grounded_output
    omnispatial_submission = {'id': doc['idx'], 'gt_content': grounded_output, 'pred_parsed': pred_letter, 'pred': response, 'sub_task': doc['sub_task'], 'is_correct': flag}
    return {key_name: omnispatial_submission}

def blink_aggregate_results(results: List[Dict]):
    total_samples = len(results)
    total_correct = 0
    for sample in results:
        if sample['is_correct']:
            total_correct += 1
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy
