import re
from typing import Any, Dict, Optional, Tuple, Union
from .prompt import BINARY_JUDGE_PROMPT, COMPARATIVE_JUDGE_PROMPT, CORRECTNESS_JUDGE_PROMPT

class JudgePromptBuilder:

    @staticmethod
    def build_binary_prompt(question: str, answer: str, prediction: str, output_format: str='0/1', custom_prompt: Optional[str]=None, **kwargs) -> str:
        if custom_prompt:
            return custom_prompt.format(question=question, answer=answer, pred=prediction, prediction=prediction, **kwargs)
        positive, negative = ('1', '0') if output_format == '0/1' or output_format == '1/0' else ('Yes', 'No')
        return BINARY_JUDGE_PROMPT.format(question=question, answer=answer, prediction=prediction, positive=positive, negative=negative)

    @staticmethod
    def build_comparative_prompt(question: str, response1: str, response2: str, context: Optional[str]=None, score_range: Tuple[int, int]=(1, 10), custom_prompt: Optional[str]=None, evaluation_instruction: Optional[str]=None, **kwargs) -> str:
        if custom_prompt:
            return custom_prompt.format(question=question, response1=response1, response2=response2, context=context or '', **kwargs)
        context_section = f'[Context]\n{context}\n\n' if context else ''
        if not evaluation_instruction:
            evaluation_instruction = f'Please provide scores from {score_range[0]} to {score_range[1]}.'
        return COMPARATIVE_JUDGE_PROMPT.format(question=question, response1=response1, response2=response2, context_section=context_section, min_score=score_range[0], max_score=score_range[1], evaluation_instruction=evaluation_instruction)

    @staticmethod
    def build_correctness_prompt(question: str, answer: str, prediction: str, output_format: str='yes/no', **kwargs) -> str:
        positive, negative = ('Yes', 'No') if output_format == 'yes/no' else ('1', '0')
        return CORRECTNESS_JUDGE_PROMPT.format(question=question, answer=answer, prediction=prediction, positive=positive, negative=negative)

class ResponseParser:

    @staticmethod
    def parse_binary_response(response: str, output_format: str='0/1') -> Union[int, bool]:
        response = response.strip().lower()
        if output_format == '0/1' or output_format == '1/0':
            if any((pattern in response for pattern in ['1', '[1]', 'score: 1', 'answer: 1'])):
                return 1
            else:
                return 0
        else:
            return response == 'yes' or response.startswith('yes')

    @staticmethod
    def parse_score_response(response: str, score_range: Optional[Tuple[float, float]]=None) -> float:
        try:
            numbers = re.findall('-?\\d+(?:\\.\\d+)?', response)
            if numbers:
                score = float(numbers[0])
                if score_range:
                    score = max(score_range[0], min(score, score_range[1]))
                return score
        except Exception:
            pass
        return score_range[0] if score_range else 0.0

    @staticmethod
    def parse_comparative_response(response: str) -> Tuple[float, float]:
        try:
            lines = response.strip().split('\n')
            if lines:
                score_line = lines[0]
                score_line = score_line.replace(',', ' ').replace(';', ' ')
                scores = re.findall('-?\\d+(?:\\.\\d+)?', score_line)
                if len(scores) >= 2:
                    return (float(scores[0]), float(scores[1]))
        except Exception:
            pass
        return (-1.0, -1.0)

    @staticmethod
    def parse_json_response(response: str) -> Dict[str, Any]:
        try:
            json_match = re.search('\\{.*\\}', response, re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group())
        except Exception:
            pass
        return {}
