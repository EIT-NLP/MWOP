import re

def parse_reasoning_model_answer(model_answer: str) -> str:
    cleaned = re.sub('<think>.*?</think>', '', model_answer, flags=re.DOTALL).strip()
    answer_match = re.search('<answer>\\s*(.*?)\\s*</answer>', cleaned, re.DOTALL)
    boxed_answer_match = re.search('\\\\boxed\\{\\s*(.*?)\\s*\\}', cleaned, re.DOTALL)
    if answer_match:
        return answer_match.group(1)
    elif boxed_answer_match:
        return boxed_answer_match.group(1)
    else:
        return cleaned
