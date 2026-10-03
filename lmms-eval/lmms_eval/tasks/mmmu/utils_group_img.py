import ast
import json
import os
import re
from collections import defaultdict
from loguru import logger as eval_logger
from PIL import Image, ImageDraw, ImageFont
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.mmmu_mcq_utils import get_multi_choice_info as shared_get_multi_choice_info
from lmms_eval.tasks._task_utils.mmmu_mcq_utils import parse_mmmu_multi_choice_response

def add_order_label(image, label, font_size=40):
    draw = ImageDraw.Draw(image)
    font_path = os.path.join(__file__, os.pardir, 'arial.ttf')
    font = ImageFont.truetype(font_path, font_size)
    text_width = text_height = font_size
    label_background_margin = 10
    label_background_size = (text_width + 2 * label_background_margin, text_height + 2 * label_background_margin)
    label_background_position = (0, 0)
    draw.rectangle([label_background_position, (label_background_position[0] + label_background_size[0], label_background_position[1] + label_background_size[1])], fill='white')
    label_position = (label_background_margin, label_background_margin)
    draw.text(label_position, label, font=font, fill='black')
    return image

def resize_image_height(image, fixed_size):
    width, height = image.size
    new_size = (int(width * fixed_size / height), fixed_size)
    return image.resize(new_size, Image.Resampling.LANCZOS)

def concatenate_images_horizontal(image_list):
    widths, heights = zip(*(i.size for i in image_list))
    total_width = sum(widths)
    max_height = max(heights)
    assert all((height == max_height for height in heights))
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in image_list:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im

def resize_image_width(image, fixed_size):
    width, height = image.size
    new_size = (fixed_size, int(height * fixed_size / width))
    return image.resize(new_size, Image.Resampling.LANCZOS)

def concatenate_images_vertical(image_list):
    widths, heights = zip(*(i.size for i in image_list))
    total_height = sum(heights)
    max_width = max(widths)
    assert all((width == max_width for width in widths))
    new_im = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for im in image_list:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im

def process_images_horizontal(original_images, size):
    images = []
    for i, img in enumerate(original_images):
        img_resized = resize_image_height(img, fixed_size=size)
        img_labeled = add_order_label(img_resized, f'[{i + 1}]')
        images.append(img_labeled)
    return concatenate_images_horizontal(images)

def process_images_vertical(original_images, size):
    images = []
    for i, img in enumerate(original_images):
        img_resized = resize_image_width(img, fixed_size=size)
        img_labeled = add_order_label(img_resized, f'[{i + 1}]')
        images.append(img_labeled)
    return concatenate_images_vertical(images)

def process_images(images, size=1008):
    concat_horizontal = process_images_horizontal(images, size)
    concat_vertical = process_images_vertical(images, size)
    hw, hh = concat_horizontal.size
    vw, vh = concat_vertical.size
    ha = hw / hh
    va = vh / vw
    if ha > va:
        return concat_vertical
    else:
        return concat_horizontal
MULTI_CHOICE_PROMPT = "Answer with the option's letter from the given choices directly."
OPEN_ENDED_PROMPT = 'Answer the question using a single word or phrase.'

def replace_images_tokens(input_string):
    return input_string

def parse_options(options):
    option_letters = [chr(ord('A') + i) for i in range(len(options))]
    choices_str = '\n'.join([f'({option_letter}) {option}' for option_letter, option in zip(option_letters, options)])
    return choices_str

def construct_prompt(doc):
    question = doc['question']
    if doc['question_type'] == 'multiple-choice':
        parsed_options = parse_options(ast.literal_eval(doc['options']))
        question = f'{question}\n{parsed_options}\n\n{MULTI_CHOICE_PROMPT}'
    else:
        question = f'{question}\n{OPEN_ENDED_PROMPT}'
    return question

def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)

def mmmu_doc_to_visual(doc):
    prompt = construct_prompt(doc)
    image_tokens = re.findall('<image \\d+>', prompt)
    image_tokens = sorted(list(set([image_token.strip('<>').replace(' ', '_') for image_token in image_tokens])))
    visual = [doc[image_token].convert('RGB') for image_token in image_tokens]
    visual = process_images(visual)
    return [visual]

def mmmu_process_results(doc, results):
    pred = results[0]
    if doc['question_type'] == 'multiple-choice':
        index2ans, all_choices = get_multi_choice_info(ast.literal_eval(doc['options']))
        parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    else:
        parsed_pred = parse_open_response(pred)
    id = doc['id']
    mmmu_acc = {'id': id, 'subdomain': extract_subset_name(doc['id']), 'question_type': doc['question_type'], 'answer': doc['answer'], 'parsed_pred': parsed_pred}
    return {'mmmu_acc': mmmu_acc, 'submission': {id: pred}}

def extract_subset_name(input_string):
    split = input_string.split('_')[0]
    pattern = re.compile(f'^{split}_(.+?)_\\d+$')
    match = pattern.search(input_string)
    if match:
        return match.group(1)
    else:
        raise ValueError(f'No match found in "{input_string}"')

def mmmu_test_aggregate_results_for_submission(results, args):
    path = generate_submission_file('mmmu_test_for_submission.json', args)
    with open(path, 'w') as f:
        json.dump(results, f)
    eval_logger.info(f'Results saved to {path}.')

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
        printable_results['Overall-' + domain] = {'num': int(in_domain_data_num), 'acc': round(in_domain_ins_acc, 3)}
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {'num': int(cat_results['num_example']), 'acc': round(cat_results['acc'], 3)}
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results['Overall'] = {'num': sum([cat_results['num_example'] for cat_results in evaluation_result.values()]), 'acc': round(all_ins_acc, 3)}
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
        pred_i = sample['parsed_pred']
        if sample['question_type'] == 'multiple-choice':
            correct = eval_multi_choice(gold_i, pred_i)
        else:
            correct = eval_open(gold_i, pred_i)
        if correct:
            judge_dict[sample['id']] = 'Correct'
            pred_correct += 1
        else:
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
