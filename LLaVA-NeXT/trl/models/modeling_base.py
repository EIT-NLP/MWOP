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

import json
import logging
import os
from copy import deepcopy
import torch
import torch.nn as nn
from accelerate import PartialState
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HFValidationError, LocalEntryNotFoundError, RepositoryNotFoundError
from safetensors.torch import load_file as safe_load_file
from transformers import PreTrainedModel
from ..import_utils import is_npu_available, is_peft_available, is_transformers_greater_than, is_xpu_available
if is_peft_available():
    from peft import PeftConfig, PeftModel, PeftModelForCausalLM, PeftModelForSeq2SeqLM, PromptLearningConfig, get_peft_model, prepare_model_for_kbit_training
if is_transformers_greater_than('4.33.0'):
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
else:
    from transformers.deepspeed import is_deepspeed_zero3_enabled
LAYER_PATTERNS = ['transformer.h.{layer}', 'model.decoder.layers.{layer}', 'gpt_neox.layers.{layer}', 'model.layers.{layer}']

class PreTrainedModelWrapper(nn.Module):
    transformers_parent_class = None
    supported_args = None
    supported_modules = ('v_head',)
    supported_rm_modules = ('score',)
    supported_pretrained_model_architectures = PreTrainedModel if not is_peft_available() else (PreTrainedModel, PeftModelForCausalLM, PeftModelForSeq2SeqLM)

    def __init__(self, pretrained_model=None, score_module=None, supports_rm_adapter=False, rm_adapter_name=None, **kwargs):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        self.prepare_inputs_for_generation = pretrained_model.prepare_inputs_for_generation
        self.is_loaded_in_8bit = getattr(pretrained_model, 'is_loaded_in_8bit', False)
        self.is_loaded_in_4bit = getattr(pretrained_model, 'is_loaded_in_4bit', False)
        self.is_sequential_parallel = False
        if hasattr(pretrained_model, 'gradient_checkpointing_disable'):
            self.gradient_checkpointing_disable = pretrained_model.gradient_checkpointing_disable
        if hasattr(pretrained_model, 'gradient_checkpointing_enable'):
            self.gradient_checkpointing_enable = pretrained_model.gradient_checkpointing_enable
        self.supports_rm_adapter = supports_rm_adapter
        self.rm_adapter_name = rm_adapter_name
        self.policy_adapter_name = 'default'
        if score_module is not None:
            self.score = score_module

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if kwargs is not None:
            peft_config = kwargs.pop('peft_config', None)
            reward_adapter = kwargs.pop('reward_adapter', None)
            reward_adapter_name = kwargs.pop('reward_adapter_name', 'reward_adapter')
            is_trainable = kwargs.pop('is_trainable', False)
            trl_model_args, pretrained_kwargs, peft_quantization_kwargs = cls._split_kwargs(kwargs)
            token = pretrained_kwargs.get('token', None)
        else:
            peft_config = None
            is_trainable = False
            trl_model_args = {}
            pretrained_kwargs = {}
            peft_quantization_kwargs = {}
            token = None
        if reward_adapter is not None and (not isinstance(reward_adapter, str)):
            raise ValueError('The `reward_adapter` argument should be a string representing the name of local path or the Hub id to the Reward Modeling adapter.')
        is_peft_model = False
        current_device = cls._get_current_device()
        if isinstance(pretrained_model_name_or_path, str):
            is_loaded_in_8bit = pretrained_kwargs['load_in_8bit'] if 'load_in_8bit' in pretrained_kwargs else False
            is_loaded_in_4bit = pretrained_kwargs['load_in_4bit'] if 'load_in_4bit' in pretrained_kwargs else False
        else:
            is_loaded_in_8bit = getattr(pretrained_model_name_or_path, 'is_loaded_in_8bit', False)
            is_loaded_in_4bit = getattr(pretrained_model_name_or_path, 'is_loaded_in_4bit', False)
        if (is_loaded_in_8bit or is_loaded_in_4bit) and 'device_map' not in pretrained_kwargs:
            logging.warning('The `device_map` argument is not provided. We will override the device_map argument. to set the entire model on the current device. If you want to set the model on multiple devices, please provide a custom `device_map` argument.')
            pretrained_kwargs['device_map'] = {'': current_device}
        if is_peft_available() and peft_config is not None and (not isinstance(peft_config, PeftConfig)):
            raise ValueError('The `peft_config` argument should be an instance of `peft.PeftConfig` class.')
        if isinstance(pretrained_model_name_or_path, str):
            if is_peft_available():
                try:
                    remote_adapter_config = hf_hub_download(pretrained_model_name_or_path, 'adapter_config.json', token=token)
                except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError, RepositoryNotFoundError):
                    remote_adapter_config = None
            else:
                remote_adapter_config = None
            local_adapter_present = os.path.exists(os.path.join(pretrained_model_name_or_path, 'adapter_config.json'))
            if (local_adapter_present or remote_adapter_config is not None) and is_peft_available():
                if peft_config is not None:
                    logging.warning(f'`peft_config` argument ignored since a peft config file was found in {pretrained_model_name_or_path}')
                if local_adapter_present:
                    trained_adapter_config = PeftConfig.from_pretrained(pretrained_model_name_or_path)
                else:
                    remote_adapter_dir = os.path.dirname(remote_adapter_config)
                    trained_adapter_config = PeftConfig.from_pretrained(remote_adapter_dir)
                pretrained_model = cls.transformers_parent_class.from_pretrained(trained_adapter_config.base_model_name_or_path, *model_args, **pretrained_kwargs)
                pretrained_model = PeftModel.from_pretrained(pretrained_model, pretrained_model_name_or_path, is_trainable=is_trainable)
                logging.info('Trained peft adapter loaded')
            else:
                pretrained_model = cls.transformers_parent_class.from_pretrained(pretrained_model_name_or_path, *model_args, **pretrained_kwargs)
                if peft_config is not None:
                    if is_loaded_in_8bit or is_loaded_in_4bit:
                        pretrained_model = prepare_model_for_kbit_training(pretrained_model, **peft_quantization_kwargs)
                    pretrained_model = get_peft_model(pretrained_model, peft_config)
                    logging.info('peft adapter initialised')
        elif isinstance(pretrained_model_name_or_path, cls.supported_pretrained_model_architectures):
            pretrained_model = pretrained_model_name_or_path
            if peft_config is not None and isinstance(pretrained_model, PreTrainedModel):
                if is_loaded_in_8bit or is_loaded_in_4bit:
                    pretrained_model = prepare_model_for_kbit_training(pretrained_model, **peft_quantization_kwargs)
                pretrained_model = get_peft_model(pretrained_model, peft_config)
                logging.info('peft adapter initialised')
        else:
            raise ValueError(f'pretrained_model_name_or_path should be a string or a PreTrainedModel, but is {type(pretrained_model_name_or_path)}')
        if is_peft_available():
            if isinstance(pretrained_model, PeftModel):
                is_peft_model = True
                if hasattr(pretrained_model, 'active_peft_config') and isinstance(pretrained_model.active_peft_config, PromptLearningConfig):
                    raise ValueError('PromptLearningConfig is not supported for PPO training.')
        if not is_peft_model and reward_adapter is not None:
            raise ValueError('reward_adapter can only be used with a PeftModel. ')
        elif is_peft_model and reward_adapter is not None:
            score_module = cls.add_and_load_reward_modeling_adapter(pretrained_model, reward_adapter, reward_adapter_name, token=token)
            multi_adapter_args = {'score_module': score_module, 'supports_rm_adapter': True, 'rm_adapter_name': reward_adapter_name}
        else:
            multi_adapter_args = {'supports_rm_adapter': False}
        model = cls(pretrained_model, **multi_adapter_args, **trl_model_args)
        is_resuming_training = True
        if isinstance(pretrained_model_name_or_path, str):
            safe_filename = os.path.join(pretrained_model_name_or_path, 'model.safetensors')
            filename = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
            sharded_index_filename = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin.index.json')
            safe_sharded_index_filename = os.path.join(pretrained_model_name_or_path, 'model.safetensors.index.json')
            is_sharded = False
            use_safe = os.path.exists(safe_filename)
            if not (os.path.exists(filename) or os.path.exists(safe_filename)):
                filename, files_to_download, is_sharded, is_resuming_training = cls._get_checkpoint_from_hub(pretrained_model, pretrained_model_name_or_path, sharded_index_filename, token=token)
                if filename is None and files_to_download is None:
                    safe_filename, files_to_download, is_sharded, is_resuming_training = cls._get_checkpoint_from_hub(pretrained_model, pretrained_model_name_or_path, safe_sharded_index_filename, token=token, model_name='model.safetensors', model_index_name='model.safetensors.index.json')
                    use_safe = True
                else:
                    use_safe = False
            loading_func = safe_load_file if use_safe else torch.load
            load_kwargs = {} if use_safe else {'map_location': 'cpu'}
            if is_resuming_training:
                if is_sharded:
                    state_dict = {}
                    for shard_file in files_to_download:
                        filename = hf_hub_download(pretrained_model_name_or_path, shard_file, token=token)
                        state_dict.update(loading_func(filename, **load_kwargs))
                else:
                    state_dict = loading_func(filename if not use_safe else safe_filename, **load_kwargs)
        else:
            state_dict = pretrained_model_name_or_path.state_dict()
        model.is_peft_model = is_peft_model
        model.current_device = current_device
        if is_resuming_training:
            model.post_init(state_dict=state_dict)
        return model

    @classmethod
    def _get_checkpoint_from_hub(cls, pretrained_model, pretrained_model_name_or_path, index_filename, token=None, model_name='pytorch_model.bin', model_index_name='pytorch_model.bin.index.json'):
        files_to_download = None
        filename = None
        is_resuming_training = True
        is_sharded = False
        try:
            filename = hf_hub_download(pretrained_model_name_or_path, model_name, token=token)
        except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError, RepositoryNotFoundError):
            if os.path.exists(index_filename):
                index_file_name = index_filename
            else:
                try:
                    index_file_name = hf_hub_download(pretrained_model_name_or_path, model_index_name, token=token)
                except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError, RepositoryNotFoundError):
                    is_resuming_training = False
                    logging.warning(f"A {type(pretrained_model)} model is loaded from '{pretrained_model_name_or_path}', and no v_head weight is found. This IS expected if you are not resuming PPO training.")
            if is_resuming_training:
                with open(index_file_name, 'r') as f:
                    index = json.load(f)
                files_to_download = set()
                for k, v in index['weight_map'].items():
                    if any([module in k for module in cls.supported_modules]):
                        files_to_download.add(v)
                is_sharded = True
        return (filename, files_to_download, is_sharded, is_resuming_training)

    @classmethod
    def _get_current_device(cls):
        state = PartialState()
        if is_xpu_available():
            return f'xpu:{state.local_process_index}'
        elif is_npu_available():
            return f'npu:{state.local_process_index}'
        else:
            return state.local_process_index if torch.cuda.is_available() else 'cpu'

    @classmethod
    def _split_kwargs(cls, kwargs):
        check_peft_kwargs = False
        if is_peft_available():
            from peft import prepare_model_for_kbit_training
            check_peft_kwargs = True
        supported_kwargs = {}
        unsupported_kwargs = {}
        peft_kwargs = {}
        for key, value in kwargs.items():
            if key in cls.supported_args:
                supported_kwargs[key] = value
            else:
                unsupported_kwargs[key] = value
            if check_peft_kwargs:
                if key in prepare_model_for_kbit_training.__code__.co_varnames:
                    peft_kwargs[key] = value
                    if key in unsupported_kwargs:
                        unsupported_kwargs.pop(key)
        return (supported_kwargs, unsupported_kwargs, peft_kwargs)

    @classmethod
    def add_and_load_reward_modeling_adapter(cls, pretrained_model, adapter_model_id, adapter_name='reward_model_adapter', token=None):
        pretrained_model.load_adapter(adapter_model_id, adapter_name, is_trainable=False)
        pretrained_model.train()
        filename = os.path.join(adapter_model_id, 'adapter_model.bin')
        safe_loading = False
        if not os.path.exists(filename):
            try:
                local_filename = hf_hub_download(adapter_model_id, 'adapter_model.bin', token=token)
            except:
                filename = os.path.join(adapter_model_id, 'adapter_model.safetensors')
                safe_loading = True
                if not os.path.exists(filename):
                    try:
                        local_filename = hf_hub_download(adapter_model_id, 'adapter_model.safetensors', token=token)
                    except:
                        raise ValueError('Could not find adapter model in the Hub, make sure you have the correct adapter model id.')
                else:
                    local_filename = filename
        else:
            local_filename = filename
        loading_func = safe_load_file if safe_loading else torch.load
        load_kwargs = {} if safe_loading else {'map_location': 'cpu'}
        adapter_state_dict = loading_func(local_filename, **load_kwargs)
        for score_name_candidate in cls.supported_rm_modules:
            if any([score_name_candidate in name for name in adapter_state_dict.keys()]):
                score_name = score_name_candidate
                break
        score_dict = {}
        for name, param in adapter_state_dict.items():
            if score_name in name:
                key_name = '.'.join(name.split('.')[-1:])
                score_dict[key_name] = param.to(cls._get_current_device())
        num_labels, hidden_dim = score_dict['weight'].shape
        has_bias = any(['bias' in name for name in adapter_state_dict.keys()])
        score = nn.Linear(hidden_dim, num_labels, bias=has_bias).to(device=cls._get_current_device(), dtype=pretrained_model.dtype)
        score.load_state_dict(score_dict)
        for param in score.parameters():
            param.requires_grad = False
        return score

    def push_to_hub(self, *args, **kwargs):
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        state_dict = kwargs.get('state_dict')
        if state_dict is None:
            state_dict = self.state_dict()
            kwargs['state_dict'] = state_dict
        if self.is_peft_model:
            save_path = args[0]
            save_path = os.path.join(save_path, 'pytorch_model.bin')
            torch.save(state_dict, save_path)
            _ = kwargs.pop('state_dict', None)
        return self.pretrained_model.save_pretrained(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        raise NotImplementedError

    def post_init(self, *args, **kwargs):
        raise NotImplementedError

    def compute_reward_score(self, input_ids, attention_mask=None, **kwargs):
        if not self.supports_rm_adapter:
            raise ValueError('This model does not support reward modeling adapter.')
        self.pretrained_model.set_adapter(self.rm_adapter_name)
        self.pretrained_model.eval()
        with torch.no_grad():
            base_model_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True, **kwargs)
            last_hidden_states = base_model_output.hidden_states[-1]
            scores = self.score(last_hidden_states)
        self.pretrained_model.set_adapter(self.policy_adapter_name)
        self.pretrained_model.eval()
        return scores

def create_reference_model(model: PreTrainedModelWrapper, num_shared_layers: int=None, pattern: str=None) -> PreTrainedModelWrapper:
    if is_deepspeed_zero3_enabled():
        raise ValueError('DeepSpeed ZeRO-3 is enabled and is not compatible with `create_reference_model()`. Please instantiate your reference model directly with `AutoCausalLM.from_pretrained()`.')
    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)
    if num_shared_layers is None:
        for param_name in parameter_names:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False
        return ref_model.eval()
    if pattern is not None:
        pattern = pattern.format(layer=num_shared_layers)
    else:
        for pattern_candidate in LAYER_PATTERNS:
            pattern_candidate = pattern_candidate.format(layer=num_shared_layers)
            if any([pattern_candidate in name for name in parameter_names]):
                pattern = pattern_candidate
                break
    if pattern is None:
        raise ValueError('Layer pattern could not be matched.')
    shared_param_list = []
    unshared_param_list = []
    shared_parameter = True
    for name, param in model.named_parameters():
        if pattern in name:
            shared_parameter = False
        if shared_parameter:
            shared_param_list.append(name)
        else:
            unshared_param_list.append(name)
    for param_name in shared_param_list:
        param = model.get_parameter(param_name)
        param.requires_grad = False
        ref_param = ref_model.get_parameter(param_name)
        ref_param = param
    for param_name in unshared_param_list:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
    if pattern is not None and len(unshared_param_list) == 0:
        logging.warning('Pattern passed or found, but no layers matched in the model. Check for a typo.')
    return ref_model.eval()
