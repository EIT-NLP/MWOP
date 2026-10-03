# Copyright 2023 DDPO-pytorch authors (Kevin Black), The HuggingFace Team, metric-space. All rights reserved.
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

import contextlib
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.utils import convert_state_dict_to_diffusers
from ..core import randn_tensor
from ..import_utils import is_peft_available
if is_peft_available():
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

@dataclass
class DDPOPipelineOutput(object):
    images: torch.Tensor
    latents: torch.Tensor
    log_probs: torch.Tensor

@dataclass
class DDPOSchedulerOutput(object):
    latents: torch.Tensor
    log_probs: torch.Tensor

class DDPOStableDiffusionPipeline(object):

    def __call__(self, *args, **kwargs) -> DDPOPipelineOutput:
        raise NotImplementedError

    def scheduler_step(self, *args, **kwargs) -> DDPOSchedulerOutput:
        raise NotImplementedError

    @property
    def unet(self):
        raise NotImplementedError

    @property
    def vae(self):
        raise NotImplementedError

    @property
    def tokenizer(self):
        raise NotImplementedError

    @property
    def scheduler(self):
        raise NotImplementedError

    @property
    def text_encoder(self):
        raise NotImplementedError

    @property
    def autocast(self):
        raise NotImplementedError

    def set_progress_bar_config(self, *args, **kwargs):
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        raise NotImplementedError

    def get_trainable_layers(self, *args, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

def _left_broadcast(input_tensor, shape):
    input_ndim = input_tensor.ndim
    if input_ndim > len(shape):
        raise ValueError('The number of dimensions of the tensor to broadcast cannot be greater than the length of the shape to broadcast to')
    return input_tensor.reshape(input_tensor.shape + (1,) * (len(shape) - input_ndim)).broadcast_to(shape)

def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(timestep.device)
    alpha_prod_t_prev = torch.where(prev_timestep.cpu() >= 0, self.alphas_cumprod.gather(0, prev_timestep.cpu()), self.final_alpha_cumprod).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = beta_prod_t_prev / beta_prod_t * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance

def scheduler_step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, eta: float=0.0, use_clipped_model_output: bool=False, generator=None, prev_sample: Optional[torch.FloatTensor]=None) -> DDPOSchedulerOutput:
    if self.num_inference_steps is None:
        raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(prev_timestep.cpu() >= 0, self.alphas_cumprod.gather(0, prev_timestep.cpu()), self.final_alpha_cumprod)
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)
    beta_prod_t = 1 - alpha_prod_t
    if self.config.prediction_type == 'epsilon':
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_epsilon = model_output
    elif self.config.prediction_type == 'sample':
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
    elif self.config.prediction_type == 'v_prediction':
        pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
        pred_epsilon = alpha_prod_t ** 0.5 * model_output + beta_prod_t ** 0.5 * sample
    else:
        raise ValueError(f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`')
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)
    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = eta * variance ** 0.5
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)
    if use_clipped_model_output:
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * pred_epsilon
    prev_sample_mean = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
    if prev_sample is not None and generator is not None:
        raise ValueError('Cannot pass both generator and prev_sample. Please make sure that either `generator` or `prev_sample` stays `None`.')
    if prev_sample is None:
        variance_noise = randn_tensor(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
        prev_sample = prev_sample_mean + std_dev_t * variance_noise
    log_prob = -(prev_sample.detach() - prev_sample_mean) ** 2 / (2 * std_dev_t ** 2) - torch.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(np.pi)))
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return DDPOSchedulerOutput(prev_sample.type(sample.dtype), log_prob)

@torch.no_grad()
def pipeline_step(self, prompt: Optional[Union[str, List[str]]]=None, height: Optional[int]=None, width: Optional[int]=None, num_inference_steps: int=50, guidance_scale: float=7.5, negative_prompt: Optional[Union[str, List[str]]]=None, num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents: Optional[torch.FloatTensor]=None, prompt_embeds: Optional[torch.FloatTensor]=None, negative_prompt_embeds: Optional[torch.FloatTensor]=None, output_type: Optional[str]='pil', return_dict: bool=True, callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None, callback_steps: int=1, cross_attention_kwargs: Optional[Dict[str, Any]]=None, guidance_rescale: float=0.0):
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    text_encoder_lora_scale = cross_attention_kwargs.get('scale', None) if cross_attention_kwargs is not None else None
    prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, lora_scale=text_encoder_lora_scale)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, latents)
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    all_latents = [latents]
    all_log_probs = []
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            if do_classifier_free_guidance and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
            scheduler_output = scheduler_step(self.scheduler, noise_pred, t, latents, eta)
            latents = scheduler_output.latents
            log_prob = scheduler_output.log_probs
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            if i == len(timesteps) - 1 or (i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
    if not output_type == 'latent':
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None
    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
    if hasattr(self, 'final_offload_hook') and self.final_offload_hook is not None:
        self.final_offload_hook.offload()
    return DDPOPipelineOutput(image, all_latents, all_log_probs)

class DefaultDDPOStableDiffusionPipeline(DDPOStableDiffusionPipeline):

    def __init__(self, pretrained_model_name: str, *, pretrained_model_revision: str='main', use_lora: bool=True):
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name, revision=pretrained_model_revision)
        self.use_lora = use_lora
        self.pretrained_model = pretrained_model_name
        self.pretrained_revision = pretrained_model_revision
        try:
            self.sd_pipeline.load_lora_weights(pretrained_model_name, weight_name='pytorch_lora_weights.safetensors', revision=pretrained_model_revision)
            self.use_lora = True
        except OSError:
            if use_lora:
                warnings.warn('If you are aware that the pretrained model has no lora weights to it, ignore this message. Otherwise please check the if `pytorch_lora_weights.safetensors` exists in the model folder.')
        self.sd_pipeline.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config)
        self.sd_pipeline.safety_checker = None
        self.sd_pipeline.vae.requires_grad_(False)
        self.sd_pipeline.text_encoder.requires_grad_(False)
        self.sd_pipeline.unet.requires_grad_(not self.use_lora)

    def __call__(self, *args, **kwargs) -> DDPOPipelineOutput:
        return pipeline_step(self.sd_pipeline, *args, **kwargs)

    def scheduler_step(self, *args, **kwargs) -> DDPOSchedulerOutput:
        return scheduler_step(self.sd_pipeline.scheduler, *args, **kwargs)

    @property
    def unet(self):
        return self.sd_pipeline.unet

    @property
    def vae(self):
        return self.sd_pipeline.vae

    @property
    def tokenizer(self):
        return self.sd_pipeline.tokenizer

    @property
    def scheduler(self):
        return self.sd_pipeline.scheduler

    @property
    def text_encoder(self):
        return self.sd_pipeline.text_encoder

    @property
    def autocast(self):
        return contextlib.nullcontext if self.use_lora else None

    def save_pretrained(self, output_dir):
        if self.use_lora:
            state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.sd_pipeline.unet))
            self.sd_pipeline.save_lora_weights(save_directory=output_dir, unet_lora_layers=state_dict)
        self.sd_pipeline.save_pretrained(output_dir)

    def set_progress_bar_config(self, *args, **kwargs):
        self.sd_pipeline.set_progress_bar_config(*args, **kwargs)

    def get_trainable_layers(self):
        if self.use_lora:
            lora_config = LoraConfig(r=4, lora_alpha=4, init_lora_weights='gaussian', target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'])
            self.sd_pipeline.unet.add_adapter(lora_config)
            for param in self.sd_pipeline.unet.parameters():
                if param.requires_grad:
                    param.data = param.to(torch.float32)
            return self.sd_pipeline.unet
        else:
            return self.sd_pipeline.unet

    def save_checkpoint(self, models, weights, output_dir):
        if len(models) != 1:
            raise ValueError('Given how the trainable params were set, this should be of length 1')
        if self.use_lora and hasattr(models[0], 'peft_config') and (getattr(models[0], 'peft_config', None) is not None):
            state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(models[0]))
            self.sd_pipeline.save_lora_weights(save_directory=output_dir, unet_lora_layers=state_dict)
        elif not self.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, 'unet'))
        else:
            raise ValueError(f'Unknown model type {type(models[0])}')

    def load_checkpoint(self, models, input_dir):
        if len(models) != 1:
            raise ValueError('Given how the trainable params were set, this should be of length 1')
        if self.use_lora:
            lora_state_dict, network_alphas = self.sd_pipeline.lora_state_dict(input_dir, weight_name='pytorch_lora_weights.safetensors')
            self.sd_pipeline.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=models[0])
        elif not self.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder='unet')
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f'Unknown model type {type(models[0])}')
