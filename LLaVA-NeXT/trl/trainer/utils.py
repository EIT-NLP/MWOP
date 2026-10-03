# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for the OV implementation in this repository.

import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from accelerate import PartialState
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from ..import_utils import is_peft_available, is_unsloth_available, is_xpu_available
from ..trainer.model_config import ModelConfig
if is_peft_available():
    from peft import LoraConfig, PeftConfig

class AdaptiveKLController:

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

class FixedKLController:

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    def __init__(self, response_template: Union[str, List[int]], instruction_template: Union[str, List[int]]=None, *args, mlm: bool=False, ignore_index: int=-100, **kwargs):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            self.instruction_token_ids = instruction_template
        self.response_template = response_template
        if isinstance(response_template, str):
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            self.response_token_ids = response_template
        if not self.mlm and self.instruction_template and (self.tokenizer.pad_token_id == self.tokenizer.eos_token_id):
            warnings.warn('The pad_token_id and eos_token_id values of this tokenizer are identical. If you are planning for multi-turn training, it can result in the model continuously generating questions and answers without eos token. To avoid this, set the pad_token_id to a different value.')
        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None
                for idx in np.where(batch['labels'][i] == self.response_token_ids[0])[0]:
                    if self.response_token_ids == batch['labels'][i][idx:idx + len(self.response_token_ids)].tolist():
                        response_token_ids_start_idx = idx
                if response_token_ids_start_idx is None:
                    warnings.warn(f"Could not find response key `{self.response_template}` in the following instance: {self.tokenizer.decode(batch['input_ids'][i])} This instance will be ignored in loss calculation. Note, if this happens often, consider increasing the `max_seq_length`.")
                    batch['labels'][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)
                    batch['labels'][i, :response_token_ids_end_idx] = self.ignore_index
        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []
                for assistant_idx in np.where(batch['labels'][i] == self.response_token_ids[0])[0]:
                    if self.response_token_ids == batch['labels'][i][assistant_idx:assistant_idx + len(self.response_token_ids)].tolist():
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))
                if len(response_token_ids_idxs) == 0:
                    warnings.warn(f"Could not find response key `{self.response_template}` in the following instance: {self.tokenizer.decode(batch['input_ids'][i])} This instance will be ignored in loss calculation. Note, if this happens often, consider increasing the `max_seq_length`.")
                    batch['labels'][i, :] = self.ignore_index
                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch['labels'][i] == human_token_ids[0])[0]:
                    if human_token_ids == batch['labels'][i][human_idx:human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)
                if len(human_token_ids_idxs) == 0:
                    warnings.warn(f"Could not find instruction key `{self.instruction_template}` in the following instance: {self.tokenizer.decode(batch['input_ids'][i])} This instance will be ignored in loss calculation. Note, if this happens often, consider increasing the `max_seq_length`.")
                    batch['labels'][i, :] = self.ignore_index
                if len(human_token_ids_idxs) > 0 and len(response_token_ids_idxs) > 0 and (human_token_ids_idxs[0] > response_token_ids_idxs[0]):
                    human_token_ids_idxs = [0] + human_token_ids_idxs
                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    if idx != 0:
                        batch['labels'][i, start:end] = self.ignore_index
                    else:
                        batch['labels'][i, :end] = self.ignore_index
                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch['labels'][i, human_token_ids_idxs[-1]:] = self.ignore_index
        return batch

@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = 'pt'

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        has_margin = 'margin' in features[0]
        for feature in features:
            if 'input_ids_chosen' not in feature or 'input_ids_rejected' not in feature or 'attention_mask_chosen' not in feature or ('attention_mask_rejected' not in feature):
                raise ValueError('The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`')
            features_chosen.append({'input_ids': feature['input_ids_chosen'], 'attention_mask': feature['attention_mask_chosen']})
            features_rejected.append({'input_ids': feature['input_ids_rejected'], 'attention_mask': feature['attention_mask_rejected']})
            if has_margin:
                margin.append(feature['margin'])
        batch_chosen = self.tokenizer.pad(features_chosen, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors)
        batch_rejected = self.tokenizer.pad(features_rejected, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors)
        batch = {'input_ids_chosen': batch_chosen['input_ids'], 'attention_mask_chosen': batch_chosen['attention_mask'], 'input_ids_rejected': batch_rejected['input_ids'], 'attention_mask_rejected': batch_rejected['attention_mask'], 'return_loss': True}
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch['margin'] = margin
        return batch

@dataclass
class DPODataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.startswith('prompt') and k.endswith('input_ids'):
                        if self.pad_token_id is None:
                            raise ValueError('Padding is enabled, but the tokenizer is not configured with a padding token. Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`) before calling the trainer.')
                        padding_value = self.pad_token_id
                    elif k.endswith('_attention_mask'):
                        padding_value = 0
                    elif k.startswith('chosen') or k.startswith('rejected') or 'decoder' in k:
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    if 'prompt' in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.endswith('_input_ids'):
                        if self.pad_token_id is None:
                            raise ValueError('Padding is enabled, but the tokenizer is not configured with a padding token. Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`) before calling the trainer.')
                        padding_value = self.pad_token_id
                    elif k.endswith('_labels'):
                        padding_value = self.label_pad_token_id
                    elif k.endswith('_attention_mask'):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    if 'prompt' in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith('_logps'):
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]
        return padded_batch

class ConstantLengthDataset(IterableDataset):

    def __init__(self, tokenizer, dataset, dataset_text_field=None, formatting_func=None, infinite=False, seq_length=1024, num_of_sequences=1024, chars_per_token=3.6, eos_token_id=0, shuffle=True, append_concat_token=True, add_special_tokens=True):
        self.tokenizer = tokenizer
        if tokenizer.eos_token_id is None:
            warnings.warn(f'The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id.')
        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        self.append_concat_token = append_concat_token
        self.add_special_tokens = add_special_tokens
        if formatting_func is None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:
            self.formatting_func = formatting_func
        if formatting_func is not None:
            if formatting_func.__code__.co_argcount > 1:
                warnings.warn('The passed formatting_func has more than one argument. Usually that function should have a single argument `example` which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing.')

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = ([], 0)
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn('The dataset reached end and the iterator is reset to the start.')
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, add_special_tokens=self.add_special_tokens, truncation=False)['input_ids']
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.append_concat_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i:i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {'input_ids': torch.LongTensor(example), 'labels': torch.LongTensor(example)}

class RunningMoments:

    def __init__(self, accelerator):
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24
        self.accelerator = accelerator

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        if self.accelerator.use_distributed:
            xs_mean, xs_var, xs_count = get_global_statistics(self.accelerator, xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = (xs_mean.float(), xs_var.float())
        delta = xs_mean - self.mean
        tot_count = self.count + xs_count
        new_sum = xs_var * xs_count
        old_sum = self.var * self.count + delta ** 2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum
        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count
        return (xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item())

@torch.no_grad()
def get_global_statistics(accelerator, xs: torch.Tensor, mask=None, device='cpu') -> Tuple[float, float, int]:
    xs = xs.to(accelerator.device)
    sum_and_count = torch.tensor([xs.sum(), xs.numel() if mask is None else mask.sum()], device=xs.device)
    sum_and_count = accelerator.reduce(sum_and_count)
    global_sum, count = sum_and_count
    global_mean = global_sum / count
    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    sum_var = accelerator.reduce(sum_var)
    global_var = sum_var / count
    return (global_mean.to(device), global_var.to(device), count.to(device))

def compute_accuracy(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    if np.array(predictions[:, 0] == predictions[:, 1], dtype=float).sum() > 0:
        warnings.warn(f'There are {np.array(predictions[:, 0] == predictions[:, 1]).sum()} out of {len(predictions[:, 0])} instances where the predictions for both options are equal. As a consequence the accuracy can be misleading.')
    predictions = np.argmax(predictions, axis=1)
    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    return {'accuracy': accuracy}

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int=-1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def exact_div(a, b, a_str, b_str, custom_error_message=''):
    q = a // b
    if a != q * b:
        raise ValueError(f'{custom_error_message}, {a_str}={a}, {b_str}={b}, inexact division: {a} / {b} = {a / b}')
    return q

class PerPromptStatTracker:

    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        prompts = np.array(prompts)
        rewards = np.array(rewards)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)
            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-06
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-06
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        return advantages

    def get_stats(self):
        return {k: {'mean': np.mean(v), 'std': np.std(v), 'count': len(v)} for k, v in self.stats.items()}

def neftune_post_forward_hook(module, input, output):
    if module.training:
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
        output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
    return output

def peft_module_casting_to_bf16(model):
    from peft.tuners.tuners_utils import BaseTunerLayer
    for name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            module = module.to(torch.bfloat16)
        elif isinstance(module, torch.nn.LayerNorm) or 'norm' in name:
            module = module.to(torch.float32)
        elif any((x in name for x in ['lm_head', 'embed_tokens', 'wte', 'wpe'])):
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

def trl_sanitze_kwargs_for_tagging(model, tag_names, kwargs=None):
    if is_unsloth_available():
        if hasattr(model, 'config') and getattr(model.config, 'unsloth_version', None) is not None:
            tag_names.append('unsloth')
    if kwargs is not None:
        if 'tags' not in kwargs:
            kwargs['tags'] = tag_names
        elif 'tags' in kwargs and isinstance(kwargs['tags'], list):
            kwargs['tags'].extend(tag_names)
        elif 'tags' in kwargs and isinstance(kwargs['tags'], str):
            tag_names.append(kwargs['tags'])
            kwargs['tags'] = tag_names
    return kwargs

def get_quantization_config(model_config: ModelConfig) -> Optional[BitsAndBytesConfig]:
    if model_config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=model_config.torch_dtype, bnb_4bit_quant_type=model_config.bnb_4bit_quant_type, bnb_4bit_use_double_quant=model_config.use_bnb_nested_quant)
    elif model_config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None
    return quantization_config

def get_kbit_device_map() -> Optional[Dict[str, int]]:
    if is_xpu_available():
        return {'': f'xpu:{PartialState().local_process_index}'}
    elif torch.cuda.is_available():
        return {'': PartialState().local_process_index}
    else:
        return None

def get_peft_config(model_config: ModelConfig) -> 'Optional[PeftConfig]':
    if model_config.use_peft is False:
        return None
    peft_config = LoraConfig(r=model_config.lora_r, lora_alpha=model_config.lora_alpha, lora_dropout=model_config.lora_dropout, bias='none', task_type='CAUSAL_LM', target_modules=model_config.lora_target_modules, modules_to_save=model_config.lora_modules_to_save)
    return peft_config
