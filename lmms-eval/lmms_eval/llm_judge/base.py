import abc
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from .protocol import Request, Response, ServerConfig
from .utils import JudgePromptBuilder, ResponseParser

class ServerInterface(abc.ABC):

    def __init__(self, config: Optional[ServerConfig]=None):
        self.config = config or ServerConfig(model_name='gpt-4')

    @abc.abstractmethod
    def evaluate(self, request: Request) -> Response:
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        pass

    def prepare_messages(self, request: Request) -> List[Dict[str, Any]]:
        messages = request.messages.copy()
        if self.config.system_prompt and (not any((m.get('role') == 'system' for m in messages))):
            messages.insert(0, {'role': 'system', 'content': self.config.system_prompt})
        return messages

    def evaluate_binary(self, question: str, answer: str, prediction: str, output_format: str='0/1', custom_prompt: Optional[str]=None, **kwargs) -> Dict[str, Any]:
        prompt = JudgePromptBuilder.build_binary_prompt(question=question, answer=answer, prediction=prediction, output_format=output_format, custom_prompt=custom_prompt, **kwargs)
        request = Request(messages=[{'role': 'user', 'content': prompt}], question=question, answer=answer, prediction=prediction, config=self.config)
        response = self.evaluate(request)
        parsed_result = ResponseParser.parse_binary_response(response.content, output_format)
        return {'result': parsed_result, 'raw_response': response.content, 'model': response.model_used, 'prompt': prompt, 'success': response.success}

    def evaluate_comparative(self, question: str, response1: str, response2: str, context: Optional[str]=None, score_range: Tuple[int, int]=(1, 10), custom_prompt: Optional[str]=None, images: Optional[List[Union[str, bytes]]]=None, **kwargs) -> Dict[str, Any]:
        prompt = JudgePromptBuilder.build_comparative_prompt(question=question, response1=response1, response2=response2, context=context, score_range=score_range, custom_prompt=custom_prompt, **kwargs)
        request = Request(messages=[{'role': 'user', 'content': prompt}], question=question, response1=response1, response2=response2, context=context, images=images, config=self.config)
        response = self.evaluate(request)
        scores = ResponseParser.parse_comparative_response(response.content)
        return {'scores': scores, 'raw_response': response.content, 'model': response.model_used, 'prompt': prompt, 'success': response.success}

    def evaluate_with_rubric(self, question: str, prediction: str, rubric: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        rubric_text = '\n'.join([f'- {k}: {v}' for k, v in rubric.items()])
        prompt = f'Evaluate the following response according to the given rubric.\n\nQuestion: {question}\n\nResponse: {prediction}\n\nRubric:\n{rubric_text}\n\nProvide a JSON response with scores for each rubric item.'
        request = Request(messages=[{'role': 'user', 'content': prompt}], config=self.config)
        response = self.evaluate(request)
        parsed_result = ResponseParser.parse_json_response(response.content)
        return {'scores': parsed_result, 'raw_response': response.content, 'model': response.model_used, 'prompt': prompt, 'success': response.success}

class AsyncServerInterface(ServerInterface):

    def __init__(self, config: Optional[ServerConfig]=None):
        super().__init__(config)
        self.semaphore = asyncio.Semaphore(config.max_concurrent)

    @abc.abstractmethod
    async def evaluate_async(self, request: Request) -> Response:
        pass

    async def evaluate_batch(self, requests: List[Request]) -> List[Response]:
        tasks = [self.evaluate_async(request) for request in requests]
        return await asyncio.gather(*tasks)

    def evaluate(self, request: Request) -> Response:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.evaluate_async(request))

    async def evaluate_binary_async(self, question: str, answer: str, prediction: str, output_format: str='0/1', custom_prompt: Optional[str]=None, **kwargs) -> Dict[str, Any]:
        prompt = JudgePromptBuilder.build_binary_prompt(question=question, answer=answer, prediction=prediction, output_format=output_format, custom_prompt=custom_prompt, **kwargs)
        request = Request(messages=[{'role': 'user', 'content': prompt}], question=question, config=self.config)
        response = await self.evaluate_async(request)
        parsed_result = ResponseParser.parse_binary_response(response.content, output_format)
        return {'result': parsed_result, 'raw_response': response.content, 'model': response.model_used, 'prompt': prompt, 'success': response.success}

    async def evaluate_binary_batch_async(self, questions: List[str], answers: List[str], predictions: List[str], output_format: str='0/1', custom_prompt: Optional[str]=None, **kwargs) -> List[Dict[str, Any]]:
        if not len(questions) == len(answers) == len(predictions):
            raise ValueError('All input lists must have the same length')
        tasks = []
        for q, a, p in zip(questions, answers, predictions):
            task = self.evaluate_binary_async(q, a, p, output_format, custom_prompt, **kwargs)
            tasks.append(task)
        return await asyncio.gather(*tasks)

    async def evaluate_comparative_async(self, question: str, response1: str, response2: str, context: Optional[str]=None, score_range: Tuple[int, int]=(1, 10), custom_prompt: Optional[str]=None, images: Optional[List[Union[str, bytes]]]=None, **kwargs) -> Dict[str, Any]:
        prompt = JudgePromptBuilder.build_comparative_prompt(question=question, response1=response1, response2=response2, context=context, score_range=score_range, custom_prompt=custom_prompt, **kwargs)
        request = Request(messages=[{'role': 'user', 'content': prompt}], question=question, response1=response1, response2=response2, context=context, images=images, config=self.config)
        response = await self.evaluate_async(request)
        scores = ResponseParser.parse_comparative_response(response.content)
        return {'scores': scores, 'raw_response': response.content, 'model': response.model_used, 'prompt': prompt, 'success': response.success}

    async def evaluate_comparative_batch_async(self, questions: List[str], responses1: List[str], responses2: List[str], contexts: Optional[List[Optional[str]]]=None, score_range: Tuple[int, int]=(1, 10), custom_prompt: Optional[str]=None, images_list: Optional[List[Optional[List[Union[str, bytes]]]]]=None, **kwargs) -> List[Dict[str, Any]]:
        if not len(questions) == len(responses1) == len(responses2):
            raise ValueError('Questions and responses lists must have the same length')
        if contexts is None:
            contexts = [None] * len(questions)
        if images_list is None:
            images_list = [None] * len(questions)
        tasks = []
        for q, r1, r2, ctx, imgs in zip(questions, responses1, responses2, contexts, images_list):
            task = self.evaluate_comparative_async(q, r1, r2, ctx, score_range, custom_prompt, imgs, **kwargs)
            tasks.append(task)
        return await asyncio.gather(*tasks)

    async def evaluate_with_rubric_async(self, question: str, prediction: str, rubric: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        rubric_text = '\n'.join([f'- {k}: {v}' for k, v in rubric.items()])
        prompt = f'Evaluate the following response according to the given rubric.\n\nQuestion: {question}\n\nResponse: {prediction}\n\nRubric:\n{rubric_text}\n\nProvide a JSON response with scores for each rubric item.'
        request = Request(messages=[{'role': 'user', 'content': prompt}], question=question, prediction=prediction, config=self.config)
        response = await self.evaluate_async(request)
        parsed_result = ResponseParser.parse_json_response(response.content)
        return {'scores': parsed_result, 'raw_response': response.content, 'model': response.model_used, 'prompt': prompt, 'success': response.success}
