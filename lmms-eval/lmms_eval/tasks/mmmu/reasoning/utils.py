import ast
import re
from collections import defaultdict
from pathlib import Path
import yaml
from lmms_eval.tasks._task_utils.mmmu_mcq_utils import get_multi_choice_info as shared_get_multi_choice_info
from lmms_eval.tasks._task_utils.mmmu_mcq_utils import parse_mmmu_multi_choice_response
from lmms_eval.tasks._task_utils.reasoning_utils import compute_score
SYSTEM_PROMPT = 'You are a helpful assistant. When the user asks a question, your response must include two parts: first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags.Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question.'
with open(Path(__file__).parent / 'mmmu_val_reasoning.yaml', 'r') as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if '!function' not in line:
            safe_data.append(line)
    config = yaml.safe_load(''.join(safe_data))

def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f'<image {i}>'
        query_text = '<image>'
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string

def parse_options(options):
    option_letters = [chr(ord('A') + i) for i in range(len(options))]
    choices_str = '\n'.join([f'{option_letter}. {option}' for option_letter, option in zip(option_letters, options)])
    return choices_str

def construct_prompt(doc, mc_prompt='', open_ended_prompt='', prompt_type='reasoning'):
    question = doc['question']
    if doc['question_type'] == 'multiple-choice':
        parsed_options = parse_options(ast.literal_eval(doc['options']))
        question = f'{question}\n{parsed_options}\n\n{mc_prompt}'
    else:
        question = f'{question}\n\n{open_ended_prompt}'
    return question

def mmmu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        question = construct_prompt(doc)
    else:
        question = construct_prompt(doc, lmms_eval_specific_kwargs['multiple_choice_prompt'], lmms_eval_specific_kwargs['open_ended_prompt'], lmms_eval_specific_kwargs['prompt_type'])
    if config['metadata']['interleaved_format']:
        question = replace_images_tokens(question)
    return question

def mmmu_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    config['metadata']['interleaved_format'] = True
    question = mmmu_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = mmmu_doc_to_visual(doc)
    messages = [{'role': 'user', 'content': []}]
    interleaved_content = question.split('<image>')
    for i, (image, text) in enumerate(zip(visuals, interleaved_content)):
        if text.strip() != '':
            messages[0]['content'].append({'type': 'text', 'text': text.strip()})
        messages[0]['content'].append({'type': 'image', 'url': image})
    messages[0]['content'].append({'type': 'text', 'text': interleaved_content[-1].strip()})
    return messages

def mmmu_doc_to_visual(doc):
    prompt = construct_prompt(doc)
    image_tokens = re.findall('<image \\d+>', prompt)
    image_tokens = sorted(list(set([image_token.strip('<>').replace(' ', '_') for image_token in image_tokens])))
    visual = [doc[image_token].convert('RGB') for image_token in image_tokens]
    return visual

def mmmu_process_results(doc, results):
    parsed_preds = []
    for pred in results:
        if doc['question_type'] == 'multiple-choice':
            index2ans, all_choices = get_multi_choice_info(ast.literal_eval(doc['options']))
            parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
        else:
            parsed_pred = parse_open_response(pred)
            parsed_pred = str(parsed_pred[0]) if parsed_pred else ''
        parsed_preds.append(parsed_pred)
    mmmu_submission = {doc['id']: parsed_preds[0]}
    mmmu_exact_acc = {'id': doc['id'], 'subdomain': extract_subset_name(doc['id']), 'question_type': doc['question_type'], 'answer': doc['answer'], 'parsed_pred': parsed_preds}
    return {'mmmu_acc': mmmu_exact_acc, 'mmmu_acc_pass_at_k': mmmu_exact_acc, 'submission': mmmu_submission}

def mmmu_reward_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    config['metadata']['interleaved_format'] = True
    question = mmmu_doc_to_text(doc)
    visuals = mmmu_doc_to_visual(doc)
    system_messages = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]}]
    messages = [{'role': 'user', 'content': []}]
    interleaved_content = question.split('<image>')
    for i, (image, text) in enumerate(zip(visuals, interleaved_content)):
        if text.strip() != '':
            messages[0]['content'].append({'type': 'text', 'text': text.strip()})
        messages[0]['content'].append({'type': 'image', 'url': image})
    messages[0]['content'].append({'type': 'text', 'text': interleaved_content[-1].strip()})
    messages = system_messages + messages
    return messages

def mmmu_reward_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = mmmu_doc_to_text(doc)
    extra_info = {'question': question}
    for pred in results:
        score_dict = compute_score(data_source='mmmu_val', solution_str=pred.strip(), ground_truth=doc['answer'], extra_info=extra_info)
        acc_score += score_dict['acc_score']
        format_score += score_dict.get('format_reward_score', 0.0)
    return {'acc_score': acc_score / len(results) if results else 0.0, 'format_score': format_score / len(results) if results else 0.0}

def extract_subset_name(input_string):
    split = input_string.split('_')[0]
    pattern = re.compile(f'^{split}_(.+?)_\\d+$')
    match = pattern.search(input_string)
    if match:
        return match.group(1)
    else:
        raise ValueError(f'No match found in "{input_string}"')

def mmmu_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result['subdomain']].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_mmmu(sub_eval_samples)
        metric_dict.update({'num_example': len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            else:
                pass
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum([cat_results['num_example'] for cat_results in in_domain_cat_results.values()])
        printable_results['Overall-' + domain] = {'num': int(in_domain_data_num), 'acc': round(in_domain_ins_acc, 5)}
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {'num': int(cat_results['num_example']), 'acc': round(cat_results['acc'], 5)}
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results['Overall'] = {'num': sum([cat_results['num_example'] for cat_results in evaluation_result.values()]), 'acc': round(all_ins_acc, 5)}
    print(printable_results)
    return printable_results['Overall']['acc']

def calculate_ins_level_acc(results):
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results['acc'] * cat_results['num_example']
        ins_num += cat_results['num_example']
    if ins_num == 0:
        return 0
    return acc / ins_num
DOMAIN_CAT2SUB_CAT = {'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'], 'Business': ['Accounting', 'Economics', 'Finance', 'Manage', 'Marketing'], 'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics'], 'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'], 'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'], 'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering']}

def eval_multi_choice(gold_i, pred_i):
    correct = False
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    elif gold_i == pred_i:
        correct = True
    return correct

def eval_open(gold_i, pred_i):
    correct = False
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        elif pred in norm_answers:
            if not correct:
                correct = True
            break
    return correct

def evaluate_mmmu(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample['answer']
        pred_list = sample['parsed_pred']
        correct = False
        for pred_i in pred_list:
            if sample['question_type'] == 'multiple-choice':
                correct = eval_multi_choice(gold_i, pred_i)
            else:
                correct = eval_open(gold_i, pred_i)
            if correct:
                judge_dict[sample['id']] = 'Correct'
                pred_correct += 1
                break
        if not correct:
            judge_dict[sample['id']] = 'Wrong'
    if len(samples) == 0:
        return {'acc': 0}
    return (judge_dict, {'acc': pred_correct / len(samples)})

def parse_multi_choice_response(response, all_choices, index2ans):
    return parse_mmmu_multi_choice_response(response, all_choices, index2ans)

def extract_numbers(string):
    pattern_commas = '-?\\b\\d{1,3}(?:,\\d{3})+\\b'
    pattern_scientific = '-?\\d+(?:\\.\\d+)?[eE][+-]?\\d+'
    pattern_simple = '-?(?:\\d+\\.\\d+|\\.\\d+|\\d+\\b)(?![eE][+-]?\\d+)(?![,\\d])'
    numbers_with_commas = re.findall(pattern_commas, string)
    numbers_scientific = re.findall(pattern_scientific, string)
    numbers_simple = re.findall(pattern_simple, string)
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers

def check_is_number(string):
    try:
        float(string.replace(',', ''))
        return True
    except ValueError:
        return False

def normalize_str(string):
    string = string.strip()
    is_number = check_is_number(string)
    if is_number:
        string = string.replace(',', '')
        string = float(string)
        string = round(string, 2)
        return [string]
    else:
        string = string.lower()
        if len(string) == 1:
            return [' ' + string, string + ' ']
        return [string]

def parse_open_response(response):

    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip('.').lower()
        sub_responses = re.split('\\.\\s(?=[A-Z])|\\n', response)
        indicators_of_keys = ['could be ', 'so ', 'is ', 'thus ', 'therefore ', 'final ', 'answer ', 'result ']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(['='])
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    elif len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                        shortest_key_response = resp.split(indicator)[-1].strip()
            if shortest_key_response:
                if shortest_key_response.strip() not in [':', ',', '.', '!', '?', ';', ':', "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:
            return [response]
        return key_responses
    key_responses = get_key_subresponses(response)
    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))
    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list
    pred_list = list(set(pred_list))
    return pred_list

def get_multi_choice_info(options):
    return shared_get_multi_choice_info(options)
