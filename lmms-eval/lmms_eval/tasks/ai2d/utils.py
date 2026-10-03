import re
from lmms_eval.filters.extraction import ExtendedRegexFilter

def ai2d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = (doc['question'], doc['options'])
    len_choices = len(choices)
    post_prompt = lmms_eval_specific_kwargs['post_prompt']
    pre_prompt = lmms_eval_specific_kwargs['pre_prompt']
    if lmms_eval_specific_kwargs['prompt_format'] == 'mcq':
        options = [chr(ord('A') + i) for i in range(len_choices)]
        choices_str = '\n'.join([f'{option}. {choice}' for option, choice in zip(options, choices)])
        return f'{pre_prompt}{question}\n{choices_str}{post_prompt}'
    elif lmms_eval_specific_kwargs['prompt_format'] == 'qa':
        options = '\n'.join(choices)
        return f'{pre_prompt}{question}{options}{post_prompt}'
    elif lmms_eval_specific_kwargs['prompt_format'] == 'mcq_xcomposer':
        options = [chr(ord('A') + i) for i in range(len_choices)]
        choices_str = ' '.join([f'{option}. {choice}' for option, choice in zip(options, choices)])
        return f'{pre_prompt}{question}\nContext: N/A\n{choices_str}{post_prompt}'
    else:
        raise ValueError(f"Unknown prompt format: {lmms_eval_specific_kwargs['prompt_format']}")

def ai2d_doc_to_visual(doc):
    return [doc['image'].convert('RGB')]

def ai2d_doc_to_target(doc, model_specific_target_kwargs):
    if model_specific_target_kwargs == 'mcq':
        len_choices = len(doc['options'])
        options = [chr(ord('A') + i) for i in range(len_choices)]
        return options[int(doc['answer'])]
    elif model_specific_target_kwargs == 'qa':
        return doc['options'][int(doc['answer'])]

class MultiChoiceRegexFilter(ExtendedRegexFilter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        filtered_resps = []
        for r, doc in zip(resps, docs):
            option_letter_regex = re.compile('^\\s*([A-Z])\\.')
            filtered = []
            for resp in r:
                match = option_letter_regex.match(resp)
                if match:
                    filtered.append(match.group(1))
                else:
                    filtered.append(resp)
            filtered_resps.append(filtered[0])
        return filtered_resps
