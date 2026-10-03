import inspect
import math
import os
import shutil
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import datetime
import numpy as np
from accelerate import Accelerator, DistributedType, __version__ as accelerate_version
from accelerate.data_loader import SeedableRandomSampler
from accelerate.utils import InitProcessGroupKwargs, GradientAccumulationPlugin
try:
    from accelerate.utils import DataLoaderConfiguration
except ImportError:
    DataLoaderConfiguration = None
from packaging import version
from torch.utils.data import Dataset, Sampler, DataLoader, RandomSampler
from trl.trainer import DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding
from transformers import Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations import hp_params
try:
    from transformers.integrations import plot_graphs_based_on_log_history
except ImportError:

    def plot_graphs_based_on_log_history(*args, **kwargs):
        return None
try:
    from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
except ImportError:
    from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.trainer import TRAINER_STATE_NAME, is_sagemaker_mp_enabled, get_parameter_names, has_length, ALL_LAYERNORM_LAYERS, logger, is_accelerate_available, is_datasets_available, GradientAccumulationPlugin, _is_peft_model
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import HPSearchBackend, TrainOutput, is_torch_xla_available, seed_worker, speed_metrics
from transformers.training_args import ParallelMode
from transformers.trainer_pt_utils import get_dataloader_sampler, get_length_grouped_indices as get_length_grouped_indices_hf, get_model_param_count
from transformers.trainer_pt_utils import AcceleratorConfig
from typing import List, Optional
from datetime import timedelta
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches, InitProcessGroupKwargs
if is_datasets_available():
    import datasets
from llava.utils import rank0_print

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, 'ds_id'):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any((key_match in k for key_match in keys_to_match))}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def should_only_save_mm_adapter(args) -> bool:
    if getattr(args, 'lora_enable', False):
        return False
    if getattr(args, 'tune_mm_mlp_adapter', False):
        return True
    mm_tunable_parts = getattr(args, 'mm_tunable_parts', None)
    if not mm_tunable_parts:
        return False
    tunable_parts = [part.strip() for part in mm_tunable_parts.split(',') if part.strip()]
    return len(tunable_parts) == 1 and tunable_parts[0] in {'mm_mlp_adapter', 'mm_vision_resampler'}

def split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]
    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float('inf')
    return chunks

def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult=8, generator=None):
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i:i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i:i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]
    return [i for batch in batches for i in batch]

def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    assert all((l != 0 for l in lengths)), 'Should not have zero length.'
    if all((l > 0 for l in lengths)) or all((l < 0 for l in lengths)):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i:i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i:i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]
    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]
    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))
    return [i for megabatch in megabatches for i in megabatch]

def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i:i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    return [i for megabatch in megabatches for batch in megabatch for i in batch]

def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i:i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]
    return [i for megabatch in megabatches for batch in megabatch for i in batch]

def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    assert all((l != 0 for l in lengths)), 'Should not have zero length.'
    if all((l > 0 for l in lengths)) or all((l < 0 for l in lengths)):
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i:i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i:i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]
    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]
    return [i for megabatch in megabatches for i in megabatch]

class LengthGroupedSampler(Sampler):

    def __init__(self, batch_size: int, world_size: int, lengths: Optional[List[int]]=None, generator=None, variable_length: bool=False, group_by_modality: bool=False, group_by_modality_auto: bool=False):
        if lengths is None:
            raise ValueError('Lengths must be provided.')
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.variable_length:
            assert not self.group_by_modality, 'Variable length grouping is not supported with modality grouping.'
            indices = get_variable_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        elif self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        elif self.group_by_modality_auto:
            indices = get_modality_length_grouped_indices_auto(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices_auto_single(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

class LLaVATrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args = self.args
        self.trainer_mode = getattr(args, 'trainer_mode', 'regular')
        self.zo_eps = getattr(args, 'zo_eps', 0.001)
        self.zo_num_directions = getattr(args, 'zo_num_directions', 1)
        print(f'MeZOLLaVATrainer initialized with mode: {self.trainer_mode}')
        self.projected_grad = None
        self.zo_random_seed = None
        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0
        self.batch_zo_seeds = None
        self.trainable_params = [(p[0], p[1]) for p in self.model.named_parameters() if p[1].requires_grad]
        print(f'Trainable parameters amount: {len(self.trainable_params)}')
        if not self.trainable_params:
            raise ValueError('No trainable parameters found.')
        if self.trainer_mode == 'zo':
            for param in self.model.parameters():
                param.requires_grad = False
        self.mezo_update_history = []

    def zo_perturb_parameters(self, scaling_factor=1.0):
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.trainable_params:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.device, dtype=param.dtype)
            param.data += scaling_factor * z * self.zo_eps

    def zo_forward(self, model, inputs):
        model.eval()
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
        return loss.detach()

    def zo_step(self, model, inputs):
        directions = []
        if self.batch_zo_seeds is None:
            self.batch_zo_seeds = [np.random.randint(1000000000) for _ in range(self.zo_num_directions)]
            print(f'Sampled batch seeds: {self.batch_zo_seeds}')
        for seed in self.batch_zo_seeds:
            self.zo_random_seed = seed
            self.zo_perturb_parameters(scaling_factor=1.0)
            loss_plus = self.zo_forward(model, inputs)
            self.zo_perturb_parameters(scaling_factor=-2.0)
            loss_minus = self.zo_forward(model, inputs)
            print(f'Seed {seed} loss_plus: {loss_plus} loss_minus: {loss_minus}')
            self.zo_perturb_parameters(scaling_factor=1.0)
            grad_estimate = ((loss_plus - loss_minus) / (2 * self.zo_eps)).item()
            directions.append((seed, grad_estimate))
            update_entry = {'type': 'zo_step', 'global_step': self.state.global_step, 'zo_eps': self.zo_eps, 'seed': seed}
            self.mezo_update_history.append(update_entry)
        self.zo_direction_accumulator.extend(directions)
        self.zo_accumulation_count += 1
        return loss_plus / (self.args.gradient_accumulation_steps * self.zo_num_directions)

    def zo_update(self, learning_rate):
        if len(self.zo_direction_accumulator) == 0:
            print('No accumulated directions to update.')
            return
        seed_group = {}
        for seed, grad_estimate in self.zo_direction_accumulator:
            if seed in seed_group:
                seed_group[seed] += grad_estimate / (self.args.gradient_accumulation_steps * self.zo_num_directions)
            else:
                seed_group[seed] = grad_estimate / (self.args.gradient_accumulation_steps * self.zo_num_directions)
        for seed, grad_sum in seed_group.items():
            torch.manual_seed(seed)
            for name, param in self.trainable_params:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.device, dtype=param.dtype)
                if 'bias' not in name and 'layer_norm' not in name and ('layernorm' not in name):
                    param.data -= learning_rate * (grad_sum * z + self.args.weight_decay * param.data)
                else:
                    param.data -= learning_rate * grad_sum * z
        print(f'Applied MeZO update with aggregated grad estimates from {len(self.zo_direction_accumulator)} directions (aggregated to {len(seed_group)} unique seeds).Seed group: {seed_group}')
        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0
        self.batch_zo_seeds = None
        update_entry = {'type': 'mezo_update', 'global_step': self.state.global_step, 'learning_rate': learning_rate, 'seed_group': seed_group}
        self.mezo_update_history.append(update_entry)

    def _save_mezo_state(self, output_dir: str):
        mezo_state = {'weight_decay': self.args.weight_decay, 'zo_eps': self.zo_eps, 'zo_num_directions': self.zo_num_directions, 'trainable_params_names': [name for name, _ in self.trainable_params], 'trainable_params_sizes': {name: param.size() for name, param in self.trainable_params}, 'update_history': self.mezo_update_history}
        mezo_checkpoint_path = os.path.join(output_dir, 'mezo_state.pt')
        torch.save(mezo_state, mezo_checkpoint_path)
        print(f'MeZO checkpoint saved at {mezo_checkpoint_path}')

    def save_model(self, output_dir: Optional[str]=None, _internal_call: bool=False):
        ret = super().save_model(output_dir, _internal_call)
        if self.trainer_mode == 'zo':
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            self._save_mezo_state(output_dir)
        return ret

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {'num_steps': self.args.gradient_accumulation_steps}
        grad_acc_kwargs['sync_with_dataloader'] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        rank0_print('Setting NCCL timeout to INF to avoid running errors.')
        accelerator_args = {'deepspeed_plugin': self.args.deepspeed_plugin, 'gradient_accumulation_plugin': gradient_accumulation_plugin, 'kwargs_handlers': [accelerator_kwargs]}
        accelerator_signature = inspect.signature(Accelerator.__init__)
        if 'dispatch_batches' in accelerator_signature.parameters:
            accelerator_args['dispatch_batches'] = self.args.dispatch_batches
            accelerator_args['split_batches'] = self.args.split_batches
        elif DataLoaderConfiguration is not None:
            accelerator_args['dataloader_config'] = DataLoaderConfiguration(dispatch_batches=self.args.dispatch_batches, split_batches=self.args.split_batches)
        self.accelerator = Accelerator(**accelerator_args)
        self.gather_function = self.accelerator.gather_for_metrics
        self.is_deepspeed_enabled = getattr(self.accelerator.state, 'deepspeed_plugin', None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, 'fsdp_plugin', None) is not None
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get('limit_all_gathers', fsdp_plugin.limit_all_gathers)
            if is_accelerate_available('0.23.0'):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get('activation_checkpointing', fsdp_plugin.activation_checkpointing)
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError("The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic when using FSDP.")
        if self.is_deepspeed_enabled and getattr(self.args, 'hf_deepspeed_config', None) is None:
            self.propagate_args_to_deepspeed()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(self.args.train_batch_size, world_size=self.args.world_size * self.args.gradient_accumulation_steps, lengths=lengths)
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(self.args.train_batch_size, world_size=self.args.world_size * self.args.gradient_accumulation_steps, lengths=lengths, group_by_modality=True)
        elif self.args.group_by_modality_length_auto:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(self.args.train_batch_size, world_size=self.args.world_size * self.args.gradient_accumulation_steps, lengths=lengths, group_by_modality_auto=True)
        elif self.args.group_by_varlen:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps, world_size=self.args.world_size * self.args.gradient_accumulation_steps, lengths=lengths, variable_length=True)
        else:
            return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description='training')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='training')
        dataloader_params = {'batch_size': self._train_batch_size, 'collate_fn': data_collator, 'num_workers': self.args.dataloader_num_workers, 'pin_memory': self.args.dataloader_pin_memory, 'persistent_workers': self.args.dataloader_persistent_workers}
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params['sampler'] = self._get_train_sampler()
            dataloader_params['drop_last'] = self.args.dataloader_drop_last
            dataloader_params['worker_init_fn'] = seed_worker
            dataloader_params['prefetch_factor'] = self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None
        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        return dataloader

    def create_optimizer(self):
        if self.trainer_mode == 'zo':
            dummy_param = torch.nn.Parameter(torch.zeros(1, device=self.args.device))
            self.optimizer = torch.optim.AdamW([dummy_param], lr=self.args.learning_rate)
            return self.optimizer
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if 'bias' not in name]
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper['mm_projector'] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper['vision_tower'] = self.args.mm_vision_tower_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any((module_keyword in name for module_keyword in lr_mapper))]
                optimizer_grouped_parameters = [{'params': [p for n, p in opt_model.named_parameters() if n in decay_parameters and n not in special_lr_parameters and p.requires_grad], 'weight_decay': self.args.weight_decay, '_intended_lr': self.args.learning_rate}, {'params': [p for n, p in opt_model.named_parameters() if n not in decay_parameters and n not in special_lr_parameters and p.requires_grad], 'weight_decay': 0.0, '_intended_lr': self.args.learning_rate}]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend([{'params': [p for n, p in opt_model.named_parameters() if n in decay_parameters and n in module_parameters and p.requires_grad], 'weight_decay': self.args.weight_decay, 'lr': lr, '_intended_lr': lr}, {'params': [p for n, p in opt_model.named_parameters() if n not in decay_parameters and n in module_parameters and p.requires_grad], 'weight_decay': 0.0, 'lr': lr, '_intended_lr': lr}])
            else:
                optimizer_grouped_parameters = [{'params': [p for n, p in opt_model.named_parameters() if n in decay_parameters and p.requires_grad], 'weight_decay': self.args.weight_decay}, {'params': [p for n, p in opt_model.named_parameters() if n not in decay_parameters and p.requires_grad], 'weight_decay': 0.0}]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == 'Adam8bit':
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f'skipped {module}: {skipped / 2 ** 20}M params')
                        manager.register_module_override(module, 'weight', {'optim_bits': 32})
                        logger.debug(f'bitsandbytes: will optimize {module} in fp32')
                logger.info(f'skipped: {skipped / 2 ** 20}M params')
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        ce_loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        loss = ce_loss
        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial, metrics=None):
        if should_only_save_mm_adapter(self.args):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, 'use_im_start_end', False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str]=None, state_dict=None):
        if should_only_save_mm_adapter(self.args):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        if args is None:
            args = self.args
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory
                self.model_wrapped, = release_memory(self.model_wrapped)
                self.model_wrapped = self.model
                if self.is_deepspeed_enabled:
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f'Currently training with a batch size of: {self._train_batch_size}')
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(args.max_steps % num_update_steps_per_epoch > 0)
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(f'args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}')
        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                raise ValueError('Currently --debug underflow_overflow is not supported under DP. Please use DDP (torchrun or torch.distributed.launch (deprecated)).')
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)
        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False
        if self.is_deepspeed_enabled:
            if self.trainer_mode == 'zo':
                self.optimizer = self.create_optimizer()
            else:
                self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)
        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs
            _ur = os.getenv('GC_USE_REENTRANT')
            if _ur is not None:
                gradient_checkpointing_kwargs = dict(gradient_checkpointing_kwargs or {})
                gradient_checkpointing_kwargs['use_reentrant'] = _ur.strip().lower() not in ('0', 'false', 'no', '')
                rank0_print(f"[gradient_checkpointing] GC_USE_REENTRANT={_ur!r} -> use_reentrant={gradient_checkpointing_kwargs['use_reentrant']}")
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        model = self._wrap_model(self.model_wrapped)
        use_accelerator_prepare = True if model is self.model else False
        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, 'step'):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)
        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model
        if model is not self.model:
            self.model_wrapped = model
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model))
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        logger.info('***** Running training *****')
        logger.info(f'  Num examples = {num_examples:,}')
        logger.info(f'  Num Epochs = {num_train_epochs:,}')
        logger.info(f'  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}')
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f'  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}')
        logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}')
        logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
        logger.info(f'  Total optimization steps = {max_steps:,}')
        logger.info(f'  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}')
        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None
        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0
            logger.info('  Continuing training from checkpoint, will skip to saved global_step')
            logger.info(f'  Continuing training from epoch {epochs_trained}')
            logger.info(f'  Continuing training from global step {self.state.global_step}')
            if not args.ignore_data_skip:
                logger.info(f'  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} batches in the first epoch.')
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()
        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse('0.23.0'):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    for _ in train_dataloader:
                        break
                else:
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)
        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, 'set_epoch'):
                epoch_iterator.set_epoch(epoch)
            if args.past_index >= 0:
                self._past = None
            steps_in_epoch = len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            if epoch == epochs_trained and resume_from_checkpoint is not None and (steps_trained_in_current_epoch == 0):
                self._load_rng_state(resume_from_checkpoint)
            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True
            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, 'main_input_name', 'input_ids')
                    if main_input_name not in inputs:
                        logger.warning('Tried to track the number of tokens seen, however the current model is not configured properly to know what item is the input. To fix this, add a `main_input_name` attribute to the model class you are using.')
                    else:
                        input_device = inputs[main_input_name].device
                        self.state.num_input_tokens_seen += torch.sum(self.accelerator.gather(torch.tensor(inputs[main_input_name].numel(), device=input_device, dtype=torch.int64))).item()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                if self.trainer_mode == 'zo':
                    tr_loss_step = self.zo_step(model, inputs)
                else:
                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)
                torch.cuda.empty_cache()
                if args.logging_nan_inf_filter and (not is_torch_xla_available()) and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(f'Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}')
                    tr_loss += tr_loss_step
                self.current_flos += float(self.floating_point_ops(inputs))
                is_last_step_and_steps_less_than_grad_acc = steps_in_epoch <= args.gradient_accumulation_steps and step + 1 == steps_in_epoch
                if total_batched_samples % args.gradient_accumulation_steps == 0 or is_last_step_and_steps_less_than_grad_acc:
                    if self.trainer_mode == 'zo':
                        self.zo_update(learning_rate=self._get_learning_rate())
                        self.lr_scheduler.step()
                    else:
                        if is_last_step_and_steps_less_than_grad_acc:
                            self.accelerator.gradient_state._set_sync_gradients(True)
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                _grad_norm = nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            if is_accelerate_available() and self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                                grad_norm = model.get_global_grad_norm()
                                if hasattr(grad_norm, 'item'):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()
                        model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(f"There seems to be not a single sample in your epoch_iterator, stopping training at step {self.state.global_step}! This is expected if you're using an IterableDataset and set num_steps ({max_steps}) higher than the number of available samples.")
                self.control.should_training_stop = True
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning("You enabled PyTorch/XLA debug metrics but you don't have a TPU configured. Check your training configuration if this is unexpected.")
            if self.control.should_training_stop:
                break
        if args.past_index and hasattr(self, '_past'):
            delattr(self, '_past')
        logger.info('\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n')
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            if is_torch_xla_available():
                xm.rendezvous('load_best_model_at_end')
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()
            self._load_best_model()
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)
        train_loss = self._total_loss_scalar / effective_global_step
        metrics = speed_metrics('train', start_time, num_samples=num_train_samples, num_steps=self.state.max_steps, num_tokens=num_train_tokens)
        self.store_flos()
        metrics['total_flos'] = self.state.total_flos
        metrics['train_loss'] = train_loss
        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)
        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)
        if self.args.should_save and self.state.best_model_checkpoint is not None and (self.args.save_total_limit == 1):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f'Deleting older checkpoint [{checkpoint}] due to args.save_total_limit')
                    shutil.rmtree(checkpoint)
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        self._finish_current_push()
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)
        plot_graphs_based_on_log_history(log_history=self.state.log_history, output_dir=run_dir, metrics=['train_loss'])
        return TrainOutput(self.state.global_step, train_loss, metrics)

class LLaVADPOTrainer(DPOTrainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(self.args.train_batch_size, world_size=self.args.world_size, lengths=lengths, group_by_modality=True)
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if should_only_save_mm_adapter(self.args):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, 'use_im_start_end', False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        elif self.args.lora_enable:
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            from transformers.modeling_utils import unwrap_model
            unwrapped_model = unwrap_model(model)
            self.save_my_lora_ckpt(output_dir, self.args, unwrapped_model)
        else:
            super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str]=None, state_dict=None):
        if should_only_save_mm_adapter(self.args):
            pass
        else:
            super(LLaVADPOTrainer, self)._save(output_dir, state_dict)
