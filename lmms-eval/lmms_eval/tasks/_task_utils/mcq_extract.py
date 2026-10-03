import re
from typing import List, Optional
_DEFAULT_CHOICES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
_ANSWER_PHRASES = ['the answer is', 'answer is', 'the correct answer is', 'correct answer is', 'the best answer is', 'best answer is', 'the correct option is', 'correct option is', 'the best option is', 'best option is', 'the choice is', 'choice is', 'the correct choice is', 'correct choice is', 'i choose', 'i select', 'i pick', 'my answer is', 'my choice is', '옵션', '정답은', '답은', '답:', '答案是', '答案为', '选', '答えは']
_FORMAT_PRIORITY = {'start': 10, 'end': 9, 'phrase': 7, 'parentheses': 6, 'period': 5, 'colon': 4, 'right_paren': 3, 'space': 2, 'fallback': 0}

def extract_mcq_answer(response: str, choices: Optional[List[str]]=None) -> str:
    if not response or not response.strip():
        return ''
    all_choices = choices or _DEFAULT_CHOICES
    text = response.strip()
    for char in [',', '.', '!', '?', ';', ':', "'", '"']:
        text = text.strip(char)
    text = ' ' + text + ' '
    candidates: list = []
    for ch in all_choices:
        if f'({ch})' in text:
            candidates.append((ch, text.rfind(f'({ch})'), 'parentheses'))
    for ch in all_choices:
        if f'{ch}.' in text:
            candidates.append((ch, text.rfind(f'{ch}.'), 'period'))
    for ch in all_choices:
        if f'{ch}:' in text:
            candidates.append((ch, text.rfind(f'{ch}:'), 'colon'))
    for ch in all_choices:
        if f'{ch})' in text:
            candidates.append((ch, text.rfind(f'{ch})'), 'right_paren'))
    for ch in all_choices:
        if f'{ch} ' in text:
            candidates.append((ch, text.rfind(f'{ch} '), 'space'))
    text_lower = text.lower()
    for phrase in _ANSWER_PHRASES:
        idx = text_lower.find(phrase)
        if idx != -1:
            after = idx + len(phrase)
            for ch in all_choices:
                ch_pos = text.find(ch, after)
                if ch_pos != -1:
                    candidates.append((ch, ch_pos, 'phrase'))
    stripped = text.strip()
    for ch in all_choices:
        if stripped.startswith(ch) and (len(stripped) == 1 or not stripped[1].isalpha()):
            candidates.append((ch, 0, 'start'))
    for ch in all_choices:
        if stripped.endswith(ch) and (len(stripped) == 1 or not stripped[-2].isalpha()):
            candidates.append((ch, len(text) - 1, 'end'))
    if not candidates:
        for ch in all_choices:
            if ch in text:
                candidates.append((ch, text.rfind(ch), 'fallback'))
    if not candidates:
        return ''
    candidates.sort(key=lambda x: (_FORMAT_PRIORITY.get(x[2], 0), x[1]), reverse=True)
    return candidates[0][0]
