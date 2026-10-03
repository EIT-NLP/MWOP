# Copyright 2023 DDPO-pytorch authors (Kevin Black), metric-space, The HuggingFace Team. All rights reserved.
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

import os
import warnings
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Optional, Tuple
from warnings import warn
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import whoami
from ..models import DDPOStableDiffusionPipeline
from . import BaseTrainer, DDPOConfig
from .utils import PerPromptStatTracker
logger = get_logger(__name__)
MODEL_CARD_TEMPLATE = '---\nlicense: apache-2.0\ntags:\n- trl\n- ddpo\n- diffusers\n- reinforcement-learning\n- text-to-image\n- stable-diffusion\n---\n\n# {model_name}\n\nThis is a diffusion model that has been fine-tuned with reinforcement learning to\n guide the model outputs according to a value, function, or human feedback. The model can be used for image generation conditioned with text.\n\n'

class DDPOTrainer(BaseTrainer):
    _tag_names = ['trl', 'ddpo']

    def __init__(self, config: DDPOConfig, reward_function: Callable[[torch.Tensor, Tuple[str], Tuple[Any]], torch.Tensor], prompt_function: Callable[[], Tuple[str, Any]], sd_pipeline: DDPOStableDiffusionPipeline, image_samples_hook: Optional[Callable[[Any, Any, Any], Any]]=None):
        if image_samples_hook is None:
            warn('No image_samples_hook provided; no images will be logged')
        self.prompt_fn = prompt_function
        self.reward_fn = reward_function
        self.config = config
        self.image_samples_callback = image_samples_hook
        accelerator_project_config = ProjectConfiguration(**self.config.project_kwargs)
        if self.config.resume_from:
            self.config.resume_from = os.path.normpath(os.path.expanduser(self.config.resume_from))
            if 'checkpoint_' not in os.path.basename(self.config.resume_from):
                checkpoints = list(filter(lambda x: 'checkpoint_' in x, os.listdir(self.config.resume_from)))
                if len(checkpoints) == 0:
                    raise ValueError(f'No checkpoints found in {self.config.resume_from}')
                checkpoint_numbers = sorted([int(x.split('_')[-1]) for x in checkpoints])
                self.config.resume_from = os.path.join(self.config.resume_from, f'checkpoint_{checkpoint_numbers[-1]}')
                accelerator_project_config.iteration = checkpoint_numbers[-1] + 1
        self.num_train_timesteps = int(self.config.sample_num_steps * self.config.train_timestep_fraction)
        self.accelerator = Accelerator(log_with=self.config.log_with, mixed_precision=self.config.mixed_precision, project_config=accelerator_project_config, gradient_accumulation_steps=self.config.train_gradient_accumulation_steps * self.num_train_timesteps, **self.config.accelerator_kwargs)
        is_okay, message = self._config_check()
        if not is_okay:
            raise ValueError(message)
        is_using_tensorboard = config.log_with is not None and config.log_with == 'tensorboard'
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(self.config.tracker_project_name, config=dict(ddpo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict(), init_kwargs=self.config.tracker_kwargs)
        logger.info(f'\n{config}')
        set_seed(self.config.seed, device_specific=True)
        self.sd_pipeline = sd_pipeline
        self.sd_pipeline.set_progress_bar_config(position=1, disable=not self.accelerator.is_local_main_process, leave=False, desc='Timestep', dynamic_ncols=True)
        if self.accelerator.mixed_precision == 'fp16':
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == 'bf16':
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float32
        self.sd_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)
        trainable_layers = self.sd_pipeline.get_trainable_layers()
        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        self.optimizer = self._setup_optimizer(trainable_layers.parameters() if not isinstance(trainable_layers, list) else trainable_layers)
        self.neg_prompt_embed = self.sd_pipeline.text_encoder(self.sd_pipeline.tokenizer([''] if self.config.negative_prompts is None else self.config.negative_prompts, return_tensors='pt', padding='max_length', truncation=True, max_length=self.sd_pipeline.tokenizer.model_max_length).input_ids.to(self.accelerator.device))[0]
        if config.per_prompt_stat_tracking:
            self.stat_tracker = PerPromptStatTracker(config.per_prompt_stat_tracking_buffer_size, config.per_prompt_stat_tracking_min_count)
        self.autocast = self.sd_pipeline.autocast or self.accelerator.autocast
        if hasattr(self.sd_pipeline, 'use_lora') and self.sd_pipeline.use_lora:
            unet, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
            self.trainable_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
        else:
            self.trainable_layers, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
        if self.config.async_reward_computation:
            self.executor = futures.ThreadPoolExecutor(max_workers=config.max_workers)
        if config.resume_from:
            logger.info(f'Resuming from {config.resume_from}')
            self.accelerator.load_state(config.resume_from)
            self.first_epoch = int(config.resume_from.split('_')[-1]) + 1
        else:
            self.first_epoch = 0

    def compute_rewards(self, prompt_image_pairs, is_async=False):
        if not is_async:
            rewards = []
            for images, prompts, prompt_metadata in prompt_image_pairs:
                reward, reward_metadata = self.reward_fn(images, prompts, prompt_metadata)
                rewards.append((torch.as_tensor(reward, device=self.accelerator.device), reward_metadata))
        else:
            rewards = self.executor.map(lambda x: self.reward_fn(*x), prompt_image_pairs)
            rewards = [(torch.as_tensor(reward.result(), device=self.accelerator.device), reward_metadata.result()) for reward, reward_metadata in rewards]
        return zip(*rewards)

    def step(self, epoch: int, global_step: int):
        samples, prompt_image_data = self._generate_samples(iterations=self.config.sample_num_batches_per_epoch, batch_size=self.config.sample_batch_size)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        rewards, rewards_metadata = self.compute_rewards(prompt_image_data, is_async=self.config.async_reward_computation)
        for i, image_data in enumerate(prompt_image_data):
            image_data.extend([rewards[i], rewards_metadata[i]])
        if self.image_samples_callback is not None:
            self.image_samples_callback(prompt_image_data, global_step, self.accelerator.trackers[0])
        rewards = torch.cat(rewards)
        rewards = self.accelerator.gather(rewards).cpu().numpy()
        self.accelerator.log({'reward': rewards, 'epoch': epoch, 'reward_mean': rewards.mean(), 'reward_std': rewards.std()}, step=global_step)
        if self.config.per_prompt_stat_tracking:
            prompt_ids = self.accelerator.gather(samples['prompt_ids']).cpu().numpy()
            prompts = self.sd_pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = self.stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-08)
        samples['advantages'] = torch.as_tensor(advantages).reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index].to(self.accelerator.device)
        del samples['prompt_ids']
        total_batch_size, num_timesteps = samples['timesteps'].shape
        for inner_epoch in range(self.config.train_num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=self.accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}
            perms = torch.stack([torch.randperm(num_timesteps, device=self.accelerator.device) for _ in range(total_batch_size)])
            for key in ['timesteps', 'latents', 'next_latents', 'log_probs']:
                samples[key] = samples[key][torch.arange(total_batch_size, device=self.accelerator.device)[:, None], perms]
            original_keys = samples.keys()
            original_values = samples.values()
            reshaped_values = [v.reshape(-1, self.config.train_batch_size, *v.shape[1:]) for v in original_values]
            transposed_values = zip(*reshaped_values)
            samples_batched = [dict(zip(original_keys, row_values)) for row_values in transposed_values]
            self.sd_pipeline.unet.train()
            global_step = self._train_batched_samples(inner_epoch, epoch, global_step, samples_batched)
            if not self.accelerator.sync_gradients:
                raise ValueError('Optimization step should have been performed by this point. Please check calculated gradient accumulation settings.')
        if epoch != 0 and epoch % self.config.save_freq == 0 and self.accelerator.is_main_process:
            self.accelerator.save_state()
        return global_step

    def calculate_loss(self, latents, timesteps, next_latents, log_probs, advantages, embeds):
        with self.autocast():
            if self.config.train_cfg:
                noise_pred = self.sd_pipeline.unet(torch.cat([latents] * 2), torch.cat([timesteps] * 2), embeds).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = self.sd_pipeline.unet(latents, timesteps, embeds).sample
            scheduler_step_output = self.sd_pipeline.scheduler_step(noise_pred, timesteps, latents, eta=self.config.sample_eta, prev_sample=next_latents)
            log_prob = scheduler_step_output.log_probs
        advantages = torch.clamp(advantages, -self.config.train_adv_clip_max, self.config.train_adv_clip_max)
        ratio = torch.exp(log_prob - log_probs)
        loss = self.loss(advantages, self.config.train_clip_range, ratio)
        approx_kl = 0.5 * torch.mean((log_prob - log_probs) ** 2)
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float())
        return (loss, approx_kl, clipfrac)

    def loss(self, advantages: torch.Tensor, clip_range: float, ratio: torch.Tensor):
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))

    def _setup_optimizer(self, trainable_layers_parameters):
        if self.config.train_use_8bit_adam:
            import bitsandbytes
            optimizer_cls = bitsandbytes.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW
        return optimizer_cls(trainable_layers_parameters, lr=self.config.train_learning_rate, betas=(self.config.train_adam_beta1, self.config.train_adam_beta2), weight_decay=self.config.train_adam_weight_decay, eps=self.config.train_adam_epsilon)

    def _save_model_hook(self, models, weights, output_dir):
        self.sd_pipeline.save_checkpoint(models, weights, output_dir)
        weights.pop()

    def _load_model_hook(self, models, input_dir):
        self.sd_pipeline.load_checkpoint(models, input_dir)
        models.pop()

    def _generate_samples(self, iterations, batch_size):
        samples = []
        prompt_image_pairs = []
        self.sd_pipeline.unet.eval()
        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)
        for _ in range(iterations):
            prompts, prompt_metadata = zip(*[self.prompt_fn() for _ in range(batch_size)])
            prompt_ids = self.sd_pipeline.tokenizer(prompts, return_tensors='pt', padding='max_length', truncation=True, max_length=self.sd_pipeline.tokenizer.model_max_length).input_ids.to(self.accelerator.device)
            prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]
            with self.autocast():
                sd_output = self.sd_pipeline(prompt_embeds=prompt_embeds, negative_prompt_embeds=sample_neg_prompt_embeds, num_inference_steps=self.config.sample_num_steps, guidance_scale=self.config.sample_guidance_scale, eta=self.config.sample_eta, output_type='pt')
                images = sd_output.images
                latents = sd_output.latents
                log_probs = sd_output.log_probs
            latents = torch.stack(latents, dim=1)
            log_probs = torch.stack(log_probs, dim=1)
            timesteps = self.sd_pipeline.scheduler.timesteps.repeat(batch_size, 1)
            samples.append({'prompt_ids': prompt_ids, 'prompt_embeds': prompt_embeds, 'timesteps': timesteps, 'latents': latents[:, :-1], 'next_latents': latents[:, 1:], 'log_probs': log_probs, 'negative_prompt_embeds': sample_neg_prompt_embeds})
            prompt_image_pairs.append([images, prompts, prompt_metadata])
        return (samples, prompt_image_pairs)

    def _train_batched_samples(self, inner_epoch, epoch, global_step, batched_samples):
        info = defaultdict(list)
        for i, sample in enumerate(batched_samples):
            if self.config.train_cfg:
                embeds = torch.cat([sample['negative_prompt_embeds'], sample['prompt_embeds']])
            else:
                embeds = sample['prompt_embeds']
            for j in range(self.num_train_timesteps):
                with self.accelerator.accumulate(self.sd_pipeline.unet):
                    loss, approx_kl, clipfrac = self.calculate_loss(sample['latents'][:, j], sample['timesteps'][:, j], sample['next_latents'][:, j], sample['log_probs'][:, j], sample['advantages'], embeds)
                    info['approx_kl'].append(approx_kl)
                    info['clipfrac'].append(clipfrac)
                    info['loss'].append(loss)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.trainable_layers.parameters() if not isinstance(self.trainable_layers, list) else self.trainable_layers, self.config.train_max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if self.accelerator.sync_gradients:
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = self.accelerator.reduce(info, reduction='mean')
                    info.update({'epoch': epoch, 'inner_epoch': inner_epoch})
                    self.accelerator.log(info, step=global_step)
                    global_step += 1
                    info = defaultdict(list)
        return global_step

    def _config_check(self) -> Tuple[bool, str]:
        samples_per_epoch = self.config.sample_batch_size * self.accelerator.num_processes * self.config.sample_num_batches_per_epoch
        total_train_batch_size = self.config.train_batch_size * self.accelerator.num_processes * self.config.train_gradient_accumulation_steps
        if not self.config.sample_batch_size >= self.config.train_batch_size:
            return (False, f'Sample batch size ({self.config.sample_batch_size}) must be greater than or equal to the train batch size ({self.config.train_batch_size})')
        if not self.config.sample_batch_size % self.config.train_batch_size == 0:
            return (False, f'Sample batch size ({self.config.sample_batch_size}) must be divisible by the train batch size ({self.config.train_batch_size})')
        if not samples_per_epoch % total_train_batch_size == 0:
            return (False, f'Number of samples per epoch ({samples_per_epoch}) must be divisible by the total train batch size ({total_train_batch_size})')
        return (True, '')

    def train(self, epochs: Optional[int]=None):
        global_step = 0
        if epochs is None:
            epochs = self.config.num_epochs
        for epoch in range(self.first_epoch, epochs):
            global_step = self.step(epoch, global_step)

    def create_model_card(self, path: str, model_name: Optional[str]='TRL DDPO Model') -> None:
        try:
            user = whoami()['name']
        except:
            warnings.warn('Cannot retrieve user information assuming you are running in offline mode.')
            return
        if not os.path.exists(path):
            os.makedirs(path)
        model_card_content = MODEL_CARD_TEMPLATE.format(model_name=model_name, model_id=f'{user}/{path}')
        with open(os.path.join(path, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(model_card_content)

    def _save_pretrained(self, save_directory):
        self.sd_pipeline.save_pretrained(save_directory)
        self.create_model_card(save_directory)
