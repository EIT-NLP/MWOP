from typing import Any, Callable, List, Optional, Union
import torch
from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedTokenizerFast
from ..core import set_seed
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper

class BestOfNSampler(object):

    def __init__(self, model: PreTrainedModelWrapper, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], queries_to_scores: Callable[[List[str]], List[float]], length_sampler: Any, sample_size: int=4, seed: Optional[int]=None, n_candidates: int=1, generation_config: Optional[GenerationConfig]=None) -> None:
        if seed is not None:
            set_seed(seed)
        if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            raise ValueError(f'tokenizer must be a PreTrainedTokenizer or PreTrainedTokenizerFast, got {type(tokenizer)}')
        if not isinstance(model, SUPPORTED_ARCHITECTURES):
            raise ValueError(f'model must be a PreTrainedModelWrapper, got {type(model)} - supported architectures are: {SUPPORTED_ARCHITECTURES}')
        self.model = model
        self.tokenizer = tokenizer
        self.queries_to_scores = queries_to_scores
        self.length_sampler = length_sampler
        self.gen_config = generation_config
        self.sample_size = sample_size
        self.n_candidates = n_candidates

    def generate(self, tokenized_query: Union[List[int], torch.Tensor, List[torch.Tensor], List[List[int]]], skip_special_tokens: bool=True, device: Optional[Union[str, torch.device]]=None, **generation_kwargs) -> List[List[str]]:
        queries = None
        if isinstance(tokenized_query, torch.Tensor) and tokenized_query.ndim == 1:
            queries = tokenized_query.unsqueeze(0)
        elif isinstance(tokenized_query, List):
            element_type = type(tokenized_query[0])
            if element_type == int:
                queries = torch.tensor(tokenized_query).unsqueeze(0)
            elif element_type == torch.Tensor:
                queries = [tensor.reshape((1, -1)) for tensor in tokenized_query]
            else:
                queries = [torch.tensor(query).reshape((1, -1)) for query in tokenized_query]
        result = []
        for query in queries:
            queries = query.repeat((self.sample_size, 1))
            output = self.model.generate(queries.to(device), max_new_tokens=self.length_sampler(), generation_config=self.gen_config, **generation_kwargs).squeeze()
            output = self.tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)
            scores = torch.tensor(self.queries_to_scores(output))
            output = [output[i] for i in scores.topk(self.n_candidates).indices]
            result.append(output)
        return result
