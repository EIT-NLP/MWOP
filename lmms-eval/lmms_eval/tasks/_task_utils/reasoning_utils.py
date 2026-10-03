import os
import re
from math_verify import parse, verify
from openai import OpenAI
API_KEY = os.getenv('JUDGE_API_KEY', 'YOUR_API_KEY')
BASE_URL = os.getenv('JUDGE_BASE_URL', 'https://api.openai.com/v1')
MODEL_NAME = os.getenv('JUDGE_MODEL_NAME', 'gpt-4o-mini')
USE_LLM_JUDGE = os.getenv('USE_LLM_JUDGE', 'False')
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
JUDGE_PROMPT = 'You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.\n\n# Input\nGround Truth Answer:\n```\n{answer}\n```\nModel Prediction:\n```\n{prediction}\n```\n\n# Evaluation Rules\n- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.\n- For open-ended questions:\n  * Score 1 if the prediction matches the answer semantically, it can be in different format.\n  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.\n- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.\n- Treat numerical answers as correct if they match within reasonable precision\n- For questions requiring units, both value and unit must be correct\n\n# Strict Output format\n1 or 0'
JUDGE_PROMPT_WITH_ANSWER = '\nYou are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case. You will receive the question, the ground truth answer, and the model prediction.\n\n# Input\nQuestion:\n```\n{question}\n```\n\nGround Truth Answer:\n```\n{answer}\n```\nModel Prediction:\n```\n{prediction}\n```\n\n# Evaluation Rules\n- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.\n- For open-ended questions:\n  * Score 1 if the prediction matches the answer semantically, it can be in different format.\n  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.\n- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.\n- Treat numerical answers as correct if they match within reasonable precision\n- For questions requiring units, both value and unit must be correct\n\n# Strict Output format\n1 or 0\n'

def extract_boxed_answer(predict_str: str) -> str:
    boxed_start = '\\boxed{'
    start_indices = []
    pos = 0
    while True:
        pos = predict_str.find(boxed_start, pos)
        if pos == -1:
            break
        start_indices.append(pos)
        pos += 1
    if not start_indices:
        return ''
    results = []
    for start_pos in start_indices:
        brace_count = 0
        pos = start_pos + len(boxed_start) - 1
        while pos < len(predict_str):
            char = predict_str[pos]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    content_start = start_pos + len(boxed_start)
                    content = predict_str[content_start:pos]
                    results.append(content)
                    break
            pos += 1
    return results[-1] if results else ''

def extract_anwser_tag(predict_str: str) -> str:
    pattern = re.compile('<answer>(.*?)</answer>', re.DOTALL)
    match_result = re.search(pattern, predict_str)
    if match_result:
        return match_result.group(1)
    boxed_answer = extract_boxed_answer(predict_str)
    if boxed_answer:
        return boxed_answer
    lines = predict_str.strip().split('\n')
    for line in reversed(lines):
        if line.strip():
            number_match = re.search('\\b(\\d+(?:\\.\\d+)?)\\b(?:\\s*\\.?\\s*$)', line)
            if number_match:
                return number_match.group(1)
    return ''

def format_reward(predict_str: str) -> float:
    think_answer_pattern = re.compile('<think>.*</think>.*<answer>.*</answer>', re.DOTALL)
    if re.fullmatch(think_answer_pattern, predict_str):
        return 1.0
    analysis_answer_pattern = re.compile('<analysis>.*</analysis>.*<answer>.*</answer>', re.DOTALL)
    if re.fullmatch(analysis_answer_pattern, predict_str):
        return 1.0
    if extract_boxed_answer(predict_str):
        return 1.0
    if len(predict_str.strip()) > 50:
        has_math = bool(re.search('[=\\+\\-\\*/\\(\\)\\[\\]\\\\]', predict_str))
        has_answer = bool(extract_anwser_tag(predict_str))
        if has_math and has_answer:
            return 0.8
    return 0.0

def simple_parse(predict_str: str) -> str:
    if predict_str.endswith('.'):
        predict_str = predict_str[:-1]
    return predict_str.strip()

def parse_mcq(predict_str: str) -> str:
    if not predict_str or predict_str.strip() == '':
        return ''
    response = predict_str.strip()
    for char in [',', '.', '!', '?', ';', ':', "'", '"']:
        response = response.strip(char)
    response = ' ' + response + ' '
    all_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    candidates = []
    for choice in all_choices:
        if f'({choice})' in response:
            candidates.append((choice, response.rfind(f'({choice})'), 'parentheses'))
    for choice in all_choices:
        if f'{choice}.' in response:
            candidates.append((choice, response.rfind(f'{choice}.'), 'period'))
    for choice in all_choices:
        if f'{choice}:' in response:
            candidates.append((choice, response.rfind(f'{choice}:'), 'colon'))
    for choice in all_choices:
        if f'{choice})' in response:
            candidates.append((choice, response.rfind(f'{choice})'), 'right_paren'))
    for choice in all_choices:
        if f'{choice} ' in response:
            candidates.append((choice, response.rfind(f'{choice} '), 'space'))
    for choice in all_choices:
        if f'{choice}-' in response:
            candidates.append((choice, response.rfind(f'{choice}-'), 'dash'))
    for choice in all_choices:
        if f'{choice}_' in response:
            candidates.append((choice, response.rfind(f'{choice}_'), 'underscore'))
    for choice in all_choices:
        if f'{choice}=' in response:
            candidates.append((choice, response.rfind(f'{choice}='), 'equals'))
    answer_phrases = ['the answer is', 'answer is', 'the correct answer is', 'correct answer is', 'the answer', 'answer', 'correct answer', 'the correct answer', 'the best answer is', 'best answer is', 'the best answer', 'best answer', 'the option is', 'option is', 'the correct option is', 'correct option is', 'the choice is', 'choice is', 'the correct choice is', 'correct choice is', 'i choose', 'i select', 'i pick', 'my answer is', 'my choice is']
    for phrase in answer_phrases:
        if phrase in response.lower():
            phrase_start = response.lower().find(phrase)
            for choice in all_choices:
                choice_pos = response.find(choice, phrase_start)
                if choice_pos != -1:
                    candidates.append((choice, choice_pos, 'phrase'))
    for choice in all_choices:
        if response.strip().startswith(choice):
            candidates.append((choice, 0, 'start'))
    for choice in all_choices:
        if response.strip().endswith(choice):
            candidates.append((choice, len(response) - 1, 'end'))
    for i, choice in enumerate(all_choices):
        if f'{i + 1}. {choice}' in response:
            candidates.append((choice, response.rfind(f'{i + 1}. {choice}'), 'numbered'))
    if not candidates:
        for choice in all_choices:
            if choice in response:
                candidates.append((choice, response.rfind(choice), 'fallback'))
    if candidates:
        format_priority = {'start': 10, 'end': 9, 'numbered': 8, 'phrase': 7, 'parentheses': 6, 'period': 5, 'colon': 4, 'right_paren': 3, 'space': 2, 'dash': 1, 'underscore': 1, 'equals': 1, 'fallback': 0}
        candidates.sort(key=lambda x: (format_priority[x[2]], -x[1]), reverse=True)
        return candidates[0][0]
    return ''

def relax_exact_match(predict_str: str, ground_truth: str, relax_portion: float=0.9) -> float:
    if parse_mcq(ground_truth) in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        predict_str = parse_mcq(predict_str)
        if predict_str.lower().strip() == parse_mcq(ground_truth).lower().strip():
            return 1.0
        return 0.0
    if predict_str in ground_truth and len(predict_str) >= relax_portion * len(ground_truth):
        return 1.0
    if ground_truth in predict_str and len(ground_truth) >= relax_portion * len(predict_str):
        return 1.0
    return 1.0 if predict_str.strip() == ground_truth.strip() else 0.0

def llm_as_judge_sync(predict_str, ground_truth, extra_info):
    if extra_info is not None and 'question' in extra_info:
        prompt = JUDGE_PROMPT_WITH_ANSWER.format(question=extra_info['question'], answer=ground_truth, prediction=predict_str)
    else:
        prompt = JUDGE_PROMPT.format(answer=ground_truth, prediction=predict_str)
    payload = {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}], 'max_tokens': 5, 'model': MODEL_NAME}
    response = client.chat.completions.create(**payload)
    try:
        score = int(response.choices[0].message.content)
    except Exception:
        score = 0
    return score

def acc_reward(predict_str, ground_truth, extra_info=None, format_reward_score=0.0, solution_str=None):
    predict_str = simple_parse(predict_str)
    gt = simple_parse(ground_truth)
    acc_score = relax_exact_match(predict_str, gt)
    if acc_score == 0.0:
        try:
            gold = parse(gt)
            pred = parse(predict_str)
            acc_score = int(verify(gold, pred))
        except Exception:
            acc_score = 0.0
    if acc_score == 0.0 and USE_LLM_JUDGE == 'True':
        acc_score = llm_as_judge_sync(predict_str, ground_truth, extra_info)
    if acc_score == 0.0 and USE_LLM_JUDGE == 'True' and (format_reward_score == 0.0) and (solution_str is not None) and (len(solution_str) < 500):
        acc_score = llm_as_judge_sync(solution_str, ground_truth, extra_info)
    return acc_score

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    format_score = 0.1
    format_reward_score = format_reward(solution_str)
    extracted_answer = extract_anwser_tag(solution_str).strip()
    acc_score = acc_reward(extracted_answer, ground_truth, extra_info, format_reward_score, solution_str)
    predict_str = simple_parse(extracted_answer)
    gt = simple_parse(ground_truth)
    score = (1.0 - format_score) * acc_score + format_score * format_reward_score
    score_dict = {'score': score, 'acc_score': acc_score, 'format_reward_score': format_reward_score, 'predict_str': predict_str, 'ground_truth': gt}
    return score_dict
DEFAULT_REASONING_SYSTEM_PROMPT = 'You are a helpful assistant. When user asks a question, your response must include two parts: first, reasoning process enclosed in <analysis>...</analysis> tags, then final answer enclosed in <answer>...</answer> tags.Please provide a clear, concise response within <answer> </answer> tags that directly addresses to question.'

def make_reasoning_doc_to_messages(doc_to_visual_fn, doc_to_text_fn, system_prompt=None):
    prompt = system_prompt or DEFAULT_REASONING_SYSTEM_PROMPT

    def _doc_to_messages(doc, lmms_eval_specific_kwargs=None):
        question = doc_to_text_fn(doc, lmms_eval_specific_kwargs)
        visuals = doc_to_visual_fn(doc)
        system_messages = [{'role': 'system', 'content': [{'type': 'text', 'text': prompt}]}]
        user_content = []
        for visual in visuals:
            user_content.append({'type': 'image', 'url': visual})
        user_content.append({'type': 'text', 'text': question.strip()})
        return system_messages + [{'role': 'user', 'content': user_content}]
    return _doc_to_messages

def make_reasoning_process_results(data_source, doc_to_text_fn, gt_key='answer', metrics_prefix='', extra_info_fn=None):

    def _process_results(doc, results):
        question = doc_to_text_fn(doc, None)
        ground_truth = str(doc[gt_key])
        extra_info = {'question': question}
        if extra_info_fn:
            extra_info.update(extra_info_fn(doc))
        acc_score = 0
        fmt_score = 0
        for pred in results:
            score_dict = compute_score(data_source=data_source, solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
            acc_score += score_dict['acc_score']
            fmt_score += score_dict.get('format_reward_score', 0.0)
        n = len(results) or 1
        return {f'{metrics_prefix}acc_score': acc_score / n, f'{metrics_prefix}format_score': fmt_score / n}
    return _process_results
