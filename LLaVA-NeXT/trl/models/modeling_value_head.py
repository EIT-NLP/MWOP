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

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from .modeling_base import PreTrainedModelWrapper

class ValueHead(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, 'summary_dropout_prob'):
            summary_dropout_prob = kwargs.pop('summary_dropout_prob', 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        if hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        if hasattr(config, 'word_embed_proj_dim'):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, 'is_encoder_decoder'):
            if config.is_encoder_decoder and hasattr(config, 'decoder'):
                if hasattr(config.decoder, 'hidden_size'):
                    hidden_size = config.decoder.hidden_size
        self.summary = nn.Linear(hidden_size, 1)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        output = self.summary(output)
        return output

class AutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ['lm_head', 'embed_out']
    supported_args = ('summary_dropout_prob', 'v_head_initializer_range', 'v_head_init_strategy')

    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)
        if not any((hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings)):
            raise ValueError('The model does not have a language model head, please use a model that has one.')
        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)
        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        initializer_range = kwargs.pop('v_head_initializer_range', 0.2)
        init_strategy = kwargs.pop('v_head_init_strategy', None)
        if init_strategy is None:
            pass
        elif init_strategy == 'normal':
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, **kwargs):
        kwargs['output_hidden_states'] = True
        kwargs['past_key_values'] = past_key_values
        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == 'PREFIX_TUNING':
            kwargs.pop('past_key_values')
        base_model_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss
        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)
        value = self.v_head(last_hidden_state).squeeze(-1)
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()
        return (lm_logits, loss, value)

    def generate(self, *args, **kwargs):
        return self.pretrained_model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state_dict = {}
        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f'v_head.{k}'] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        setattr(self.pretrained_model, 'v_head', self.v_head)
        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def post_init(self, state_dict):
        for k in list(state_dict.keys()):
            if 'v_head.' in k:
                state_dict[k.replace('v_head.', '')] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict
        if hasattr(self.pretrained_model, 'hf_device_map'):
            if 'cpu' in self.pretrained_model.hf_device_map.values() or 'disk' in self.pretrained_model.hf_device_map.values():
                raise ValueError('The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models.')
            first_device = list(set(self.pretrained_model.hf_device_map.values()))[0]
            self.v_head = self.v_head.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output
            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True

class AutoModelForSeq2SeqLMWithValueHead(PreTrainedModelWrapper):
    transformers_parent_class = AutoModelForSeq2SeqLM
    lm_head_namings = ['lm_head', 'embed_out', 'output_projection']
    supported_args = ('summary_dropout_prob', 'v_head_initializer_range', 'v_head_init_strategy')

    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)
        self.is_encoder_decoder = True
        if not self._has_lm_head():
            raise ValueError('The model does not have a language model head, please use a model that has one.')
        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)
        self._init_weights(**v_head_kwargs)

    def _has_lm_head(self):
        for name, module in self.pretrained_model.named_modules():
            if any((attribute in name for attribute in self.lm_head_namings)):
                return True
        return False

    def post_init(self, state_dict):
        for k in list(state_dict.keys()):
            if 'v_head.' in k:
                state_dict[k.replace('v_head.', '')] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict
        if hasattr(self.pretrained_model, 'hf_device_map'):
            if 'cpu' in self.pretrained_model.hf_device_map.values() or 'disk' in self.pretrained_model.hf_device_map.values():
                raise ValueError('The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models.')
            for name, module in self.pretrained_model.named_modules():
                if any((attribute in name for attribute in self.lm_head_namings)):
                    lm_head_device = module.weight.device
                    break
            self.v_head = self.v_head.to(lm_head_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
                return new_output
            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True

    def state_dict(self, *args, **kwargs):
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state_dict = {}
        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f'v_head.{k}'] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        setattr(self.pretrained_model, 'v_head', self.v_head)
        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def _init_weights(self, **kwargs):
        initializer_range = kwargs.pop('v_head_initializer_range', 0.2)
        init_strategy = kwargs.pop('v_head_init_strategy', None)
        if init_strategy is None:
            pass
        elif init_strategy == 'normal':
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, **kwargs):
        kwargs['past_key_values'] = past_key_values
        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == 'PREFIX_TUNING':
            kwargs.pop('past_key_values')
        base_model_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
        last_hidden_state = base_model_output.decoder_hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss
        value = self.v_head(last_hidden_state).squeeze(-1)
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()
        return (lm_logits, loss, value)

    def generate(self, *args, **kwargs):
        return self.pretrained_model.generate(*args, **kwargs)
