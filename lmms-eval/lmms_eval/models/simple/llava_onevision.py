import copy
import json
import logging
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union
import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video
warnings.filterwarnings('ignore')
eval_logger = logging.getLogger('lmms-eval')
torch.backends.cuda.matmul.allow_tf32 = True
try:
    from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
except ImportError as e:
    eval_logger.debug(f'LLaVA is not installed. Please install LLaVA to use this model.\nError: {e}')
if version.parse(torch.__version__) >= version.parse('2.1.2'):
    best_fit_attn_implementation = 'sdpa'
else:
    best_fit_attn_implementation = 'eager'

@register_model('llava_onevision')
class Llava_OneVision(lmms):

    def __init__(self, pretrained: str='lmms-lab/llava-onevision-qwen2-7b-ov', truncation: Optional[bool]=True, device: Optional[str]='cuda:0', batch_size: Optional[Union[int, str]]=1, model_name: Optional[str]=None, attn_implementation: Optional[str]=best_fit_attn_implementation, device_map: Optional[str]='cuda:0', conv_template: Optional[str]='qwen_1_5', use_cache: Optional[bool]=True, truncate_context: Optional[bool]=False, customized_config: Optional[str]=None, max_frames_num: Optional[int]=32, mm_spatial_pool_stride: Optional[int]=2, mm_spatial_pool_mode: Optional[str]='bilinear', token_strategy: Optional[str]='single', video_decode_backend: str='decord', ablation_config_path: Optional[str]=None, **kwargs) -> None:
        super().__init__()
        assert kwargs == {}, f'Unexpected kwargs: {kwargs}'
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        model_parallel_device_maps = {'auto', 'balanced', 'balanced_low_0', 'sequential'}
        if accelerator.num_processes > 1:
            self._device = torch.device(f'cuda:{accelerator.local_process_index}')
            self.device_map = f'cuda:{accelerator.local_process_index}'
        elif accelerator.num_processes == 1 and device_map in model_parallel_device_maps:
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f'cuda:{accelerator.local_process_index}')
            self.device_map = f'cuda:{accelerator.local_process_index}'
        llava_model_args = {'multimodal': True}
        if customized_config is not None:
            llava_model_args['customized_config'] = customized_config
        if attn_implementation is not None:
            llava_model_args['attn_implementation'] = attn_implementation
        if 'use_flash_attention_2' in kwargs:
            llava_model_args['use_flash_attention_2'] = kwargs['use_flash_attention_2']
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)
        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend
        overwrite_config = {}
        overwrite_config['mm_spatial_pool_stride'] = self.mm_spatial_pool_stride
        overwrite_config['mm_spatial_pool_mode'] = self.mm_spatial_pool_mode
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)
        llava_model_args['overwrite_config'] = overwrite_config
        try:
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        except TypeError:
            llava_model_args.pop('multimodal', None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        self._config = self._model.config
        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        if ablation_config_path:
            try:
                from llava.model.language_model.ablation_registry import apply_ablation_config
                from llava.model.language_model.attention_mask_plugin import region_mask_decode_enabled
                from llava.model.language_model.head_mask_utils import head_mask_decode_enabled
                from llava.model.language_model.ffn_mask_plugin import ffn_mask_decode_enabled
            except ImportError as _e:
                raise ImportError("ablation_config_path requires LLaVA-NeXT's local Qwen2 fork (see LLaVA-NeXT/llava/model/language_model/ablation_registry.py).") from _e
            ablation_summary = apply_ablation_config(self._model, ablation_config_path)
            eval_logger.warning(f'[head-mask] Applied eval-time ablation from {ablation_config_path}: {ablation_summary.describe()}')
            eval_logger.warning('[head-mask] Hard region-mask scope: ' + ('prefill+decode (REGION_MASK_DECODE=1)' if region_mask_decode_enabled() else 'prefill-only (REGION_MASK_DECODE=0)'))
            eval_logger.warning('[head-mask] Whole-head mask scope: ' + ('prefill+decode (HEAD_MASK_DECODE=1)' if head_mask_decode_enabled() else 'prefill-only (HEAD_MASK_DECODE=0)'))
            eval_logger.warning('[head-mask] Token-scoped FFN mask scope: ' + ('prefill+decode (FFN_MASK_DECODE=1)' if ffn_mask_decode_enabled() else 'prefill-only (FFN_MASK_DECODE=0)'))
        import os as _os
        _struct_prune_cfg = _os.environ.get('STRUCT_PRUNE_CONFIG', '').strip()
        if _struct_prune_cfg:
            try:
                from llava.model.language_model.head_mask_utils import clear_zero_head_mask
                clear_zero_head_mask(self._model)
            except Exception:
                pass
            try:
                from structural_head_prune import prune_heads_from_zero_config
            except ImportError:
                import sys as _sys
                _pr = _os.path.join(_os.environ.get('PROJECT_ROOT', ''), 'scripts', 'prune')
                if _pr and _pr not in _sys.path:
                    _sys.path.insert(0, _pr)
                from structural_head_prune import prune_heads_from_zero_config
            _phase = int(_os.environ.get('STRUCT_PRUNE_PHASE', '1'))
            _prep = prune_heads_from_zero_config(self._model, _struct_prune_cfg, phase=_phase)
            eval_logger.warning(f"[struct-prune] phase={_phase}: removed {_prep['heads_removed']} heads over {_prep['layers_pruned']} layers from {_struct_prune_cfg}")
        _ffn_prune_cfg = _os.environ.get('STRUCT_FFN_PRUNE_CONFIG', '').strip()
        if _ffn_prune_cfg:
            try:
                from structural_ffn_prune import prune_ffn_from_config
            except ImportError:
                import sys as _sys
                _pr = _os.path.join(_os.environ.get('PROJECT_ROOT', ''), 'scripts', 'prune')
                if _pr and _pr not in _sys.path:
                    _sys.path.insert(0, _pr)
                from structural_ffn_prune import prune_ffn_from_config
            _fr = prune_ffn_from_config(self._model, _ffn_prune_cfg)
            eval_logger.warning(f"[ffn-prune] removed {_fr['neurons_removed']} neurons over {_fr['layers_pruned']} layers from {_ffn_prune_cfg}")
        assert self.batch_size_per_gpu == 1, 'Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue.'
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], 'Unsupported distributed type provided. Only DDP and FSDP are supported.'
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {'train_micro_batch_size_per_gpu': self.batch_size_per_gpu, 'train_batch_size': self.batch_size_per_gpu * accelerator.num_processes}
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info('Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0')
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f'Using {accelerator.num_processes} devices with data parallelism')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map in model_parallel_device_maps:
            eval_logger.info(f'Using one process with Hugging Face device map: {device_map}')
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f'Using single device: {self._device}')
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, 'accelerator'):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == 'left':
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == 'left':
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=self.rank != 0, desc='Model Responding')
        origin_image_aspect_ratio = getattr(self._config, 'image_aspect_ratio', None)
        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visual = doc_to_visual(self.task_dict[task][split][doc_id])
            if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                self._config.image_aspect_ratio = origin_image_aspect_ratio
                eval_logger.info(f'Resetting image aspect ratio to {origin_image_aspect_ratio}')
            if visual is None or visual == []:
                visual = None
                task_type = 'text'
                image_tensor = None
            else:
                if len(visual) > 1 or 'image_aspect_ratio' not in self._config.__dict__:
                    self._config.image_aspect_ratio = 'pad'
                    eval_logger.info(f'In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}')
                if 'task_type' in self.metadata and self.metadata['task_type'] == 'video' and ('sample_frames' in self.metadata):
                    assert type(visual) == list, 'sample_frames must be specified for video task'
                    sample_indices = np.linspace(0, len(visual) - 1, self.metadata['sample_frames'], dtype=int)
                    visual = [visual[i] for i in sample_indices]
                    assert len(visual) == self.metadata['sample_frames']
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                    task_type = 'video'
                elif isinstance(visual[0], PIL.Image.Image):
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                    task_type = 'image'
                elif type(visual[0]) == str:
                    image_tensor = []
                    try:
                        if self.video_decode_backend == 'decord':
                            frames = self.load_video(visual, self.max_frames_num)
                        elif self.video_decode_backend == 'pyav':
                            frames = read_video(visual[0], num_frm=self.max_frames_num)
                        frames = self._image_processor.preprocess(frames, return_tensors='pt')['pixel_values'].half().to(self._device)
                        image_tensor.append(frames)
                    except Exception as e:
                        eval_logger.error(f'Error {e} in loading video')
                        image_tensor = None
                    task_type = 'video'
            if image_tensor is not None and len(image_tensor) != 0 and (DEFAULT_IMAGE_TOKEN not in contexts):
                placeholder_count = len(visual) if isinstance(visual, list) else 1
                if task_type == 'video':
                    placeholder_count = len(frames) if self.token_strategy == 'multiple' else 1
                image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                image_tokens = ' '.join(image_tokens)
                prompts_input = image_tokens + '\n' + contexts
            else:
                prompts_input = contexts
            if 'llama_3' in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            conv.messages[-1][1] = continuation
            full_prompt = conv.get_prompt()
            full_input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
            labels = full_input_ids.clone()
            labels[0, :input_ids.shape[1]] = -100
            kwargs = {}
            if task_type == 'image':
                kwargs['image_sizes'] = [[v.size[0], v.size[1]] for v in visual] if isinstance(visual, list) else [[visual.size[0], visual.size[1]]]
            elif task_type == 'video':
                kwargs['modalities'] = ['video']
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode
            with torch.inference_mode():
                outputs = self.model(input_ids=full_input_ids, labels=labels, images=image_tensor, use_cache=True, **kwargs)
            loss = outputs['loss']
            logits = outputs['logits']
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = full_input_ids[:, input_ids.shape[1]:]
            greedy_tokens = greedy_tokens[:, input_ids.shape[1]:full_input_ids.shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        if not input or any((i is None for i in input)):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        from decord import VideoReader, cpu
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return (-len(toks), x[0])
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=self.rank != 0, desc='Model Responding')
        origin_image_aspect_ratio = getattr(self._config, 'image_aspect_ratio', None)
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]
            assert len(batched_visuals) == 1
            gen_kwargs = all_gen_kwargs[0]
            if 'until' in gen_kwargs:
                gen_kwargs.pop('until')
            question_input = []
            for visual, context in zip(batched_visuals, batched_contexts):
                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f'Resetting image aspect ratio to {origin_image_aspect_ratio}')
                if visual is None or visual == []:
                    visual = None
                    task_type = 'text'
                    placeholder_count = 0
                    image_tensor = None
                else:
                    if len(visual) > 1 or 'image_aspect_ratio' not in self._config.__dict__:
                        self._config.image_aspect_ratio = getattr(gen_kwargs, 'image_aspect_ratio', 'pad')
                        eval_logger.info(f'In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}')
                    if 'task_type' in metadata and metadata['task_type'] == 'video' and ('sample_frames' in metadata):
                        assert type(visual) == list, 'sample_frames must be specified for video task'
                        sample_indices = np.linspace(0, len(visual) - 1, metadata['sample_frames'], dtype=int)
                        visual = [visual[i] for i in sample_indices]
                        assert len(visual) == metadata['sample_frames']
                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                        task_type = 'video'
                        placeholder_count = 1
                    elif type(visual[0]) == PIL.Image.Image:
                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                        task_type = 'image'
                        placeholder_count = len(visual) if isinstance(visual, list) else 1
                    elif type(visual[0]) == str:
                        image_tensor = []
                        try:
                            if self.video_decode_backend == 'decord':
                                frames = self.load_video(visual, self.max_frames_num)
                            elif self.video_decode_backend == 'pyav':
                                frames = read_video(visual[0], num_frm=self.max_frames_num)
                            frames = self._image_processor.preprocess(frames, return_tensors='pt')['pixel_values'].half().to(self._device)
                            image_tensor.append(frames)
                        except Exception as e:
                            eval_logger.error(f'Error {e} in loading video')
                            image_tensor = None
                        task_type = 'video'
                        placeholder_count = len(frames) if self.token_strategy == 'multiple' else 1
                if image_tensor is not None and len(image_tensor) != 0 and (DEFAULT_IMAGE_TOKEN not in context):
                    "\n                    Three senarios:\n                    1. No image, and there for, no image token should be added.\n                    2. image token is already specified in the context, so we don't need to add it.\n                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.\n                    4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better\n                    "
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = ' '.join(image_tokens)
                    question = image_tokens + '\n' + context
                else:
                    question = context
                if 'llama_3' in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()
                if utils.is_json(question):
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item['value']
                        conv.append_message(role, message)
                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            if task_type == 'image':
                gen_kwargs['image_sizes'] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
            elif task_type == 'video':
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs['modalities'] = ['video']
                gen_kwargs['stopping_criteria'] = [stopping_criteria]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode
            if 'max_new_tokens' not in gen_kwargs:
                gen_kwargs['max_new_tokens'] = 1024
            if 'image_aspect_ratio' in gen_kwargs.keys():
                gen_kwargs.pop('image_aspect_ratio')
            if not gen_kwargs.get('do_sample', False):
                gen_kwargs.pop('temperature', None)
                gen_kwargs.pop('top_p', None)
                gen_kwargs.pop('top_k', None)
            try:
                with torch.inference_mode():
                    cont = self.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                raise e
            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial('generate_until', (context, gen_kwargs), text_outputs)
            pbar.update(1)
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return (-len(toks), x[0])
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=self.rank != 0, desc='Model Responding')
        origin_image_aspect_ratio = getattr(self._config, 'image_aspect_ratio', None)
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_to_text, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]
            assert len(batched_visuals) == 1
            gen_kwargs = all_gen_kwargs[0]
            if 'until' in gen_kwargs:
                gen_kwargs.pop('until')
            round_idx = 0
            batched_round_res = []
            batched_previous_round_info = None
            while True:
                question_input = []
                if round_idx != 0:
                    batched_visuals, batched_contexts, batched_terminal_singal, batched_round_res, batched_previous_round_info = list(zip(*[batched_doc_to_text[0](self.task_dict[task][split][ids], previous_output=[round_res[ids_idx] for round_res in batched_round_res], round_idx=round_idx, previous_round_info=batched_previous_round_info[ids_idx] if batched_previous_round_info is not None else None) for ids_idx, ids in enumerate(batched_doc_id)]))
                    batched_round_res = list(zip(*batched_round_res))
                    if batched_terminal_singal[0]:
                        break
                for visual, context in zip(batched_visuals, batched_contexts):
                    if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                        self._config.image_aspect_ratio = origin_image_aspect_ratio
                        eval_logger.info(f'Resetting image aspect ratio to {origin_image_aspect_ratio}')
                    if visual is None or visual == []:
                        visual = None
                        task_type = 'text'
                        placeholder_count = 0
                        image_tensor = None
                    else:
                        if len(visual) > 1 or 'image_aspect_ratio' not in self._config.__dict__:
                            self._config.image_aspect_ratio = getattr(gen_kwargs, 'image_aspect_ratio', 'pad')
                            eval_logger.info(f'In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}')
                        if 'task_type' in metadata and metadata['task_type'] == 'video' and ('sample_frames' in metadata):
                            assert type(visual) == list, 'sample_frames must be specified for video task'
                            sample_indices = np.linspace(0, len(visual) - 1, metadata['sample_frames'], dtype=int)
                            visual = [visual[i] for i in sample_indices]
                            assert len(visual) == metadata['sample_frames']
                            image_tensor = process_images(visual, self._image_processor, self._config)
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                            task_type = 'video'
                            placeholder_count = 1
                        elif type(visual[0]) == PIL.Image.Image:
                            image_tensor = process_images(visual, self._image_processor, self._config)
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                            task_type = 'image'
                            placeholder_count = len(visual) if isinstance(visual, list) else 1
                        elif type(visual[0]) == str:
                            image_tensor = []
                            try:
                                if self.video_decode_backend == 'decord':
                                    frames = self.load_video(visual, self.max_frames_num)
                                elif self.video_decode_backend == 'pyav':
                                    frames = read_video(visual[0], num_frm=self.max_frames_num)
                                frames = self._image_processor.preprocess(frames, return_tensors='pt')['pixel_values'].half().to(self._device)
                                image_tensor.append(frames)
                            except Exception as e:
                                eval_logger.error(f'Error {e} in loading video')
                                image_tensor = None
                            task_type = 'video'
                            placeholder_count = len(frames) if self.token_strategy == 'multiple' else 1
                    if image_tensor is not None and len(image_tensor) != 0 and (DEFAULT_IMAGE_TOKEN not in context):
                        "\n                        Three senarios:\n                        1. No image, and there for, no image token should be added.\n                        2. image token is already specified in the context, so we don't need to add it.\n                        3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.\n                        4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better\n                        "
                        image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                        image_tokens = ' '.join(image_tokens)
                        question = image_tokens + '\n' + context
                    else:
                        question = context
                    if 'llama_3' in self.conv_template:
                        conv = copy.deepcopy(conv_templates[self.conv_template])
                    else:
                        conv = conv_templates[self.conv_template].copy()
                    if utils.is_json(question):
                        question = json.loads(question)
                        for idx, item in enumerate(question):
                            role = conv.roles[idx % 2]
                            message = item['value']
                            conv.append_message(role, message)
                        assert len(conv.messages) % 2 == 1
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)
                    else:
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)
                if 'max_new_tokens' not in gen_kwargs:
                    gen_kwargs['max_new_tokens'] = 1024
                if 'do_sample' not in gen_kwargs:
                    gen_kwargs['do_sample'] = False
                if gen_kwargs.get('do_sample', False):
                    if 'temperature' not in gen_kwargs:
                        gen_kwargs['temperature'] = 1.0
                    if 'top_p' not in gen_kwargs:
                        gen_kwargs['top_p'] = 1.0
                if 'num_beams' not in gen_kwargs:
                    gen_kwargs['num_beams'] = 1
                input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for prompt in question_input]
                pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
                attention_masks = input_ids.ne(pad_token_ids).to(self.device)
                if task_type == 'image':
                    gen_kwargs['image_sizes'] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
                elif task_type == 'video':
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                    gen_kwargs['modalities'] = ['video']
                    gen_kwargs['stopping_criteria'] = [stopping_criteria]
                    self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                    self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode
                if 'image_aspect_ratio' in gen_kwargs.keys():
                    gen_kwargs.pop('image_aspect_ratio')
                if not gen_kwargs.get('do_sample', False):
                    gen_kwargs.pop('temperature', None)
                    gen_kwargs.pop('top_p', None)
                try:
                    with torch.inference_mode():
                        cont = self.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)
                    text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                except Exception as e:
                    raise e
                text_outputs = [response.strip() for response in text_outputs]
                batched_round_res.append(text_outputs)
                round_idx += 1
            res.extend(list(zip(*batched_round_res)))
            self.cache_hook.add_partial('generate_until_multi_round', (context, gen_kwargs), batched_round_res)
            pbar.update(1)
        res = re_ords.get_original(res)
        pbar.close()
        return res
