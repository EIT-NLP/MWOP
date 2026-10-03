from lmms_eval.tasks.videomme.utils import videomme_doc_to_text, videomme_process_results, videomme_aggregate_results
import re
import datasets
LETTER_TO_NUMBER = {'A': '1', 'B': '2', 'C': '3', 'D': '4'}
NUMBER_TO_LETTER = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}

def videomme_process_docs_number_option(dataset: datasets.Dataset) -> datasets.Dataset:
    numbers = ['1', '2', '3', '4']

    def convert_to_number_options(example):
        options = example['options']
        answer = example['answer']
        contents = []
        for opt in options:
            if '. ' in opt:
                contents.append(opt.split('. ', 1)[1])
            else:
                contents.append(opt)
        new_options = [f'{numbers[i]}. {contents[i]}' for i in range(len(contents))]
        new_answer = LETTER_TO_NUMBER.get(answer.upper(), answer)
        return {'options': new_options, 'answer': new_answer}
    return dataset.map(convert_to_number_options)

def videomme_doc_to_text_number(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get('pre_prompt', '')
    post_prompt = lmms_eval_specific_kwargs.get('post_prompt', '')
    question = doc['question']
    options = doc['options']
    options_text = '\n'.join(options)
    full_prompt = f'{pre_prompt}{question}\n{options_text}{post_prompt}'
    return full_prompt

def extract_number_answer(response):
    response = response.strip()
    patterns = ['^([1-4])[\\.\\)\\s]', '\\b([1-4])\\b']
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    if response and response[0] in '1234':
        return response[0]
    return response

def videomme_process_results_number(doc, results):
    from lmms_eval.tasks.videomme.utils import CATEGORIES, SUB_CATEGORIES, TASK_CATEGORIES
    pred = results[0] if results else ''
    pred_ans = extract_number_answer(pred)
    gt_ans = doc['answer']
    pred_letter = NUMBER_TO_LETTER.get(pred_ans, pred_ans)
    gt_letter = NUMBER_TO_LETTER.get(gt_ans, gt_ans)
    category = doc.get('domain', doc.get('category', 'Unknown'))
    if category not in CATEGORIES:
        category = CATEGORIES[-1]
    sub_category = doc.get('sub_category', 'Unknown')
    if sub_category not in SUB_CATEGORIES:
        sub_category = SUB_CATEGORIES[-1]
    task_category = doc.get('task_type', doc.get('task_category', 'Unknown'))
    if task_category not in TASK_CATEGORIES:
        task_category = TASK_CATEGORIES[-1]
    data_dict = {'question_id': doc['question_id'], 'duration': doc['duration'], 'category': category, 'sub_category': sub_category, 'task_category': task_category, 'pred_answer': pred_ans, 'answer': gt_ans}
    return {'videomme_perception_score': data_dict}
