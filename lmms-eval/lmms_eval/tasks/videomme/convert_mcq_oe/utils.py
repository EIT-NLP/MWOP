import logging
import os
import re
eval_logger = logging.getLogger('lmms-eval')
LLM_MATCH_PROMPT = 'Given a model\'s answer to a video question and the available options, determine which option matches the answer.\n\nQuestion: {question}\n\nModel\'s Answer: {model_answer}\n\nOptions:\n{options}\n\nInstructions:\n- Analyze the semantic meaning of the model\'s answer\n- Compare it with each option\n- If the model\'s answer clearly matches one option, output that letter (A, B, C, or D)\n- If the model\'s answer does NOT match any option (irrelevant, wrong, or unrelated), output "X"\n- If the model explicitly refuses to answer, says "cannot answer", or gives an empty/meaningless response, output "X"\n\nOutput format: Just output a single letter (A, B, C, D, or X), nothing else.'

def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get('pre_prompt', '')
    post_prompt = lmms_eval_specific_kwargs.get('post_prompt', '\nPlease answer the question with a short answer.')
    question = doc['question']
    full_prompt = f'{pre_prompt}{question}{post_prompt}'
    return full_prompt

def _extract_abcdx(s):
    if not s:
        return 'X'
    s = s.strip().upper()
    match = re.search('[ABCDX]', s)
    return match.group(0) if match else 'X'

def _format_options(options):
    if isinstance(options, str):
        return options
    formatted = []
    prefixes = ['A', 'B', 'C', 'D']
    for i, opt in enumerate(options):
        if i < len(prefixes):
            opt_text = opt.strip()
            if opt_text and opt_text[0] in 'ABCDabcd' and (len(opt_text) > 1) and (opt_text[1] in '.):'):
                opt_text = opt_text[2:].strip()
            formatted.append(f'{prefixes[i]}. {opt_text}')
    return '\n'.join(formatted)

def _llm_match_option(question, model_answer, options):
    try:
        from openai import OpenAI
    except ImportError:
        eval_logger.warning('openai not installed, returning X')
        return 'X'
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        eval_logger.warning('OPENAI_API_KEY not set, returning X')
        return 'X'
    try:
        client = OpenAI(api_key=api_key)
        formatted_options = _format_options(options)
        prompt = LLM_MATCH_PROMPT.format(question=question, model_answer=model_answer, options=formatted_options)
        response = client.chat.completions.create(model='gpt-4o-mini', messages=[{'role': 'user', 'content': prompt}], max_tokens=10, temperature=0)
        result = response.choices[0].message.content.strip()
        return _extract_abcdx(result)
    except Exception as e:
        eval_logger.error(f'LLM match error: {e}')
        return 'X'

def process_results(doc, results):
    pred = results[0] if results else ''
    question = doc.get('question', '')
    options = doc.get('options', [])
    matched_letter = _llm_match_option(question, pred, options)
    gt_letter = doc.get('answer', '').upper()
    if matched_letter == 'X':
        acc_score = 0.0
        llm_match_score = 0.0
    else:
        acc_score = 1.0 if matched_letter == gt_letter else 0.0
        llm_match_score = 1.0
    return {'acc_score': acc_score, 'llm_match_score': llm_match_score}
