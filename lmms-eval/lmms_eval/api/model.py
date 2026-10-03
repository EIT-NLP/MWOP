import abc
import gc
import os
from typing import List, Optional, Tuple, Type, TypeVar
import torch
import torch.nn as nn
from lmms_eval import utils
from lmms_eval.api.instance import Instance
T = TypeVar('T', bound='lmms')

class lmms(abc.ABC):
    is_simple: bool = True

    def __init__(self) -> None:
        self._rank = 0
        self._world_size = 1
        self.cache_hook = CacheHook(None)
        self.task_dict = {}

    @staticmethod
    def _resolve_system_prompt(value: str) -> str:
        if value and os.path.isfile(value):
            with open(value, 'r') as f:
                return f.read().strip()
        return value

    def _apply_system_prompt(self, messages: list, system_prompt: str) -> list:
        if not system_prompt:
            return messages
        use_list_format = True
        for msg in messages:
            if isinstance(msg.get('content'), str):
                use_list_format = False
                break
        system_content = [{'type': 'text', 'text': system_prompt}] if use_list_format else system_prompt
        if messages and messages[0].get('role') == 'system':
            messages[0]['content'] = system_content
        else:
            messages.insert(0, {'role': 'system', 'content': system_content})
        return messages

    @abc.abstractmethod
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        pass

    @abc.abstractmethod
    def generate_until(self, requests) -> List[str]:
        pass

    @abc.abstractmethod
    def generate_until_multi_round(self, requests) -> List[str]:
        pass

    def generate_visual_cot(self, requests) -> List[str]:
        raise NotImplementedError(f'{type(self).__name__} does not support Visual CoT (GtA). To run visual_cot tasks, the model must implement generate_visual_cot(). Implement generate_visual_cot() in an evaluation adapter')

    @classmethod
    def create_from_arg_string(cls: Type[T], arg_string: str, additional_config: Optional[dict]=None) -> T:
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def set_cache_hook(self, cache_hook) -> None:
        self.cache_hook = cache_hook

    def clean(self):
        for attr_name in list(vars(self)):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, nn.Module):
                delattr(self, attr_name)
        gc.collect()
        torch.cuda.empty_cache()

class CacheHook:

    def __init__(self, cachinglm=None) -> None:
        pass

    def add_partial(self, attr, req, res) -> None:
        pass
