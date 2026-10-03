import base64
import collections
import copy
import itertools
import json
import mimetypes
import os
import random
import re
from datetime import timedelta
from typing import Callable, List, Optional, Union
import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm
torch.distributed.distributed_c10d.default_pg_timeout = timedelta(hours=24)
import lmms_eval.api
import lmms_eval.api.metrics
import lmms_eval.api.registry
from lmms_eval import models
from lmms_eval.api.instance import Instance, unwrap_generation_output
from lmms_eval.api.model import lmms
from lmms_eval.api.reasoning import parse_reasoning_tags_config, strip_reasoning_tags
from lmms_eval.api.task import Task
from lmms_eval.baselines import BASELINE_REGISTRY, get_baseline_display_name, load_baseline
from lmms_eval.caching.response_cache import ResponseCache
from lmms_eval.evaluator_utils import compute_baseline_comparison, consolidate_group_results, consolidate_results, get_sample_size, get_subtask_list, get_task_list, prepare_print_tasks, print_writeout, run_task_tests
from lmms_eval.llm_judge.launcher import get_launcher
from lmms_eval.loggers.evaluation_tracker import EvaluationTracker
from lmms_eval.models.model_utils.efficiency_metrics import build_efficiency_summary
from lmms_eval.models.model_utils.usage_metrics import is_budget_exceeded, reset_usage_metrics, set_budget, set_task_context, summarize_usage_metrics
from lmms_eval.tasks import TaskManager, get_task_dict
from lmms_eval.utils import create_iterator, get_datetime_str, get_git_branch_name, get_git_commit_hash, get_lmms_eval_version_string, handle_non_serializable, hash_string, is_multimodal_content, positional_deprecated, run_task_tests, simple_parse_args_string
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tif', '.tiff')

def _enable_reentrant_filelocks() -> None:
    if getattr(_enable_reentrant_filelocks, '_patched', False):
        return
    import filelock as _fl
    import filelock._api as _fl_api
    from datasets.utils import _filelock as _datasets_filelock
    target_lock_classes = tuple((cls for cls in (getattr(_fl, 'FileLock', None), getattr(_datasets_filelock, 'FileLock', None)) if cls is not None))
    original_call = _fl_api.FileLockMeta.__call__
    _cross_class_cache: dict[str, _fl_api.BaseFileLock] = {}

    def _patched_call(cls, lock_file, *args, **kwargs):
        if cls in target_lock_classes:
            canonical = str(lock_file)
            cached = _cross_class_cache.get(canonical)
            if cached is not None:
                return cached
            kwargs.setdefault('is_singleton', True)
            if cls is getattr(_datasets_filelock, 'FileLock', None):
                unset_mode = getattr(_fl_api, '_UNSET_FILE_MODE', None)
                if kwargs.get('mode', unset_mode) == unset_mode:
                    umask = os.umask(438)
                    os.umask(umask)
                    kwargs['mode'] = 438 & ~umask
            instance = original_call(cls, lock_file, *args, **kwargs)
            _cross_class_cache[canonical] = instance
            return instance
        return original_call(cls, lock_file, *args, **kwargs)
    _fl_api.FileLockMeta.__call__ = _patched_call
    _enable_reentrant_filelocks._patched = True

def _looks_like_image_ref(value: str) -> bool:
    lowered = value.lower().strip()
    if lowered.startswith('data:image/'):
        return True
    if lowered.startswith(('http://', 'https://', 'file://')):
        return any((ext in lowered for ext in IMAGE_EXTENSIONS))
    return lowered.endswith(IMAGE_EXTENSIONS)

def _guess_image_mime(path_hint: Optional[str]) -> str:
    if path_hint:
        guessed, _ = mimetypes.guess_type(path_hint)
        if guessed and guessed.startswith('image/'):
            return guessed
    return 'image/png'

def _append_image_source(target: list[str], source: str, seen: set[str], max_items: int) -> None:
    if not source or source in seen or len(target) >= max_items:
        return
    seen.add(source)
    target.append(source)

def _extract_image_sources(value, out: list[str], seen: set[str], max_items: int=4, max_inline_bytes: int=300000) -> None:
    if len(out) >= max_items:
        return
    if isinstance(value, str):
        if _looks_like_image_ref(value):
            _append_image_source(out, value, seen, max_items)
        return
    if isinstance(value, dict):
        path_hint: Optional[str] = None
        for key in ('url', 'uri', 'path', 'image', 'image_url', 'image_path'):
            candidate = value.get(key)
            if isinstance(candidate, str):
                if key == 'path':
                    path_hint = candidate
                if _looks_like_image_ref(candidate):
                    _append_image_source(out, candidate, seen, max_items)
        raw_bytes = value.get('bytes')
        if isinstance(raw_bytes, (bytes, bytearray)) and 0 < len(raw_bytes) <= max_inline_bytes and (len(out) < max_items):
            mime = _guess_image_mime(path_hint)
            encoded = base64.b64encode(raw_bytes).decode('ascii')
            _append_image_source(out, f'data:{mime};base64,{encoded}', seen, max_items)
        for nested in value.values():
            _extract_image_sources(nested, out, seen, max_items=max_items, max_inline_bytes=max_inline_bytes)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _extract_image_sources(item, out, seen, max_items=max_items, max_inline_bytes=max_inline_bytes)
            if len(out) >= max_items:
                break

def _collect_input_media(doc: dict, request_args: list) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    _extract_image_sources(doc, sources, seen)
    for arg in request_args:
        if len(sources) >= 4:
            break
        _extract_image_sources(arg, sources, seen)
    return sources

@positional_deprecated
def simple_evaluate(model, model_args: Optional[Union[str, dict]]=None, launcher_args: Optional[Union[str, dict]]=None, tasks: Optional[List[Union[str, dict, object]]]=None, num_fewshot: Optional[int]=None, batch_size: Optional[Union[int, str]]=None, max_batch_size: Optional[int]=None, device: Optional[str]=None, use_cache: Optional[str]=None, cache_requests: bool=False, rewrite_requests_cache: bool=False, delete_requests_cache: bool=False, limit: Optional[Union[int, float]]=None, offset: int=0, bootstrap_iters: int=100000, check_integrity: bool=False, write_out: bool=False, log_samples: bool=True, evaluation_tracker: Optional[EvaluationTracker]=None, system_instruction: Optional[str]=None, apply_chat_template: bool=False, fewshot_as_multiturn: bool=False, gen_kwargs: Optional[str]=None, task_manager: Optional[TaskManager]=None, verbosity: str='INFO', predict_only: bool=False, random_seed: int=0, numpy_random_seed: int=1234, torch_random_seed: int=1234, fewshot_random_seed: int=1234, datetime_str: str=get_datetime_str(), distributed_executor_backend: str='accelerate', cli_args=None, force_simple: bool=False, repeats: int=1, baseline: Optional[str]=None, max_tokens: Optional[int]=None):
    seed_message = []
    if random_seed is not None:
        seed_message.append(f'Setting random seed to {random_seed}')
        random.seed(random_seed)
    if numpy_random_seed is not None:
        seed_message.append(f'Setting numpy seed to {numpy_random_seed}')
        np.random.seed(numpy_random_seed)
    if torch_random_seed is not None:
        seed_message.append(f'Setting torch manual seed to {torch_random_seed}')
        torch.manual_seed(torch_random_seed)
    if seed_message:
        eval_logger.info(' | '.join(seed_message))
    assert tasks != [], 'No tasks specified, or no tasks found. Please verify the task names.'
    assert distributed_executor_backend in {'accelerate', 'torchrun'}, f"Invalid distributed executor backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'."
    if gen_kwargs:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning('generation_kwargs specified through cli, these settings will be used over set parameters in yaml tasks.')
        if gen_kwargs == '':
            gen_kwargs = None
    if model_args is None:
        model_args = ''
    if launcher_args is not None:
        launcher_args = simple_parse_args_string(launcher_args)
        launcher_name = launcher_args.pop('name')
        eval_launcher = get_launcher(launcher_name)(**launcher_args)
    else:
        eval_launcher = None
    if task_manager is None:
        task_manager = TaskManager(verbosity, model_name=model)
    if isinstance(model, str):
        if model_args is None:
            model_args = ''
        lm = models.get_model(model, force_simple).create_from_arg_string(model_args, {'batch_size': batch_size, 'max_batch_size': max_batch_size, 'device': device})
    elif isinstance(model, lmms_eval.api.model.lmms):
        lm = model
    task_type = 'simple' if lm.is_simple else 'chat'
    _enable_reentrant_filelocks()
    import ssl
    if not hasattr(ssl.SSLContext, '_orig_reduce'):
        ssl.SSLContext._orig_reduce = True
        ssl.SSLContext.__reduce__ = lambda self: (ssl.SSLContext, (self.protocol,))
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        if rank == 0:
            eval_logger.info('Rank 0: loading datasets first to warm HF cache')
            task_dict = get_task_dict(tasks, task_manager, task_type)
        torch.distributed.barrier()
        if rank != 0:
            eval_logger.info(f'Rank {rank}: loading datasets from warm cache')
            task_dict = get_task_dict(tasks, task_manager, task_type)
    else:
        task_dict = get_task_dict(tasks, task_manager, task_type)

    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {**adjusted_task_dict, **{task_name: _adjust_config(task_obj)}}
            else:
                task_obj = task_dict[task_name]
                if type(task_obj) == tuple:
                    group, task_obj = task_obj
                    if task_obj is None:
                        continue
                lm.task_dict[task_name] = task_obj.dataset
                if 'generate_until' in task_obj.get_config('output_type'):
                    if gen_kwargs is not None:
                        task_obj.set_config(key='generation_kwargs', value=gen_kwargs, update=True)
                if predict_only:
                    eval_logger.info(f'Processing {task_name} in output-only mode. Metrics will not be calculated!')
                    task_obj.override_metric(metric_name='bypass')
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config('num_fewshot')) == 0:
                        eval_logger.info(f'num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored.')
                    else:
                        eval_logger.warning(f'Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}')
                        task_obj.set_config(key='num_fewshot', value=num_fewshot)
                elif (default_num_fewshot := task_obj.get_config('num_fewshot')) is None:
                    task_obj.set_config(key='num_fewshot', value=0)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                if repeats > 1:
                    default_repeats = task_obj.get_config('repeats') or 1
                    eval_logger.info(f'[Model Stability] Setting repeats={repeats} for {task_name} (was: {default_repeats})')
                    task_obj.set_config(key='repeats', value=repeats)
                adjusted_task_dict[task_name] = task_obj
        return adjusted_task_dict
    task_dict = _adjust_config(task_dict)
    if check_integrity:
        run_task_tests(task_list=tasks)
    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(model_source=model, model_args=model_args, system_instruction=system_instruction, chat_template=lm.chat_template if apply_chat_template else None, fewshot_as_multiturn=fewshot_as_multiturn)
    from lmms_eval.models.model_utils.gen_metrics import reset_logged_metrics
    reset_logged_metrics()
    reset_usage_metrics()
    if max_tokens is not None:
        set_budget(max_tokens=max_tokens)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    global_rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    response_cache = None
    if use_cache is not None:
        response_cache = ResponseCache.create(cache_root=use_cache, model=model, model_args=model_args, task_dict=task_dict, world_size=world_size, global_rank=global_rank)
    eval_succeeded = False
    try:
        results = evaluate(lm=lm, task_dict=task_dict, limit=limit, offset=offset, cache_requests=cache_requests, rewrite_requests_cache=rewrite_requests_cache, bootstrap_iters=bootstrap_iters, write_out=write_out, log_samples=True if predict_only else log_samples, system_instruction=system_instruction, apply_chat_template=apply_chat_template, fewshot_as_multiturn=fewshot_as_multiturn, verbosity=verbosity, distributed_executor_backend=distributed_executor_backend, cli_args=cli_args, eval_server_launcher=eval_launcher, response_cache=response_cache)
        eval_succeeded = True
    finally:
        if response_cache is not None:
            response_cache.finalize(success=eval_succeeded, dist_backend=distributed_executor_backend, accelerator=getattr(lm, 'accelerator', None))
    if global_rank == 0:
        from lmms_eval.models.model_utils.gen_metrics import summarize_logged_metrics
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__
        results['config'] = {'model': model_name, 'model_args': model_args}
        results['config'].update({'batch_size': batch_size, 'batch_sizes': list(lm.batch_sizes.values()) if hasattr(lm, 'batch_sizes') else [], 'device': device, 'use_cache': use_cache, 'limit': limit, 'offset': offset, 'bootstrap_iters': bootstrap_iters, 'gen_kwargs': gen_kwargs, 'random_seed': random_seed, 'numpy_seed': numpy_random_seed, 'torch_seed': torch_random_seed, 'fewshot_seed': fewshot_random_seed})
        if cli_args is not None:
            resolved = {}
            for key, value in vars(cli_args).items():
                try:
                    json.dumps(value)
                    resolved[key] = value
                except (TypeError, ValueError):
                    resolved[key] = str(value)
            results['config']['resolved_cli_args'] = resolved
        results['git_hash'] = get_git_commit_hash()
        results['git_branch'] = get_git_branch_name()
        results['lmms_eval_version'] = get_lmms_eval_version_string()
        results['date'] = datetime_str
        throughput_summary = summarize_logged_metrics()
        if throughput_summary:
            results['throughput'] = throughput_summary
        usage_summary = summarize_usage_metrics()
        results['usage'] = usage_summary
        efficiency_summary = build_efficiency_summary(results)
        if efficiency_summary:
            results['efficiency'] = efficiency_summary
        if baseline:
            baseline_display_name = get_baseline_display_name(baseline)
            for task_name in results.get('results', {}).keys():
                try:
                    baseline_scores_dict, baseline_agg = load_baseline(baseline, task_name)
                    if 'samples' in results and task_name in results['samples']:
                        current_samples = results['samples'][task_name]
                        task_config = results.get('configs', {}).get(task_name, {})
                        score_key = task_config.get('score_key', 'score')
                        current_scores = []
                        baseline_scores = []
                        for sample in current_samples:
                            doc_id = sample.get('doc_id')
                            if doc_id in baseline_scores_dict:
                                score = None
                                if score_key in sample:
                                    val = sample[score_key]
                                    if isinstance(val, (int, float)):
                                        score = float(val)
                                    elif isinstance(val, dict) and 'score' in val:
                                        score = float(val['score'])
                                if score is None:
                                    for key in sample:
                                        if key.endswith('_score') and key != score_key:
                                            val = sample[key]
                                            if isinstance(val, (int, float)):
                                                score = float(val)
                                                break
                                            elif isinstance(val, dict) and 'score' in val:
                                                score = float(val['score'])
                                                break
                                if score is not None:
                                    current_scores.append(score)
                                    baseline_scores.append(baseline_scores_dict[doc_id])
                        if current_scores and baseline_scores:
                            comparison = compute_baseline_comparison(current_scores, baseline_scores, baseline_display_name)
                            task_results = results['results'][task_name]
                            task_results['paired_baseline'] = comparison['baseline_name']
                            task_results['paired_baseline_score'] = comparison['baseline_mean'] * 100
                            task_results['paired_ci_lower'] = comparison['ci_lower'] * 100
                            task_results['paired_ci_upper'] = comparison['ci_upper'] * 100
                            task_results['paired_pvalue'] = comparison['p_value']
                            eval_logger.info(f"[Baseline] {task_name}: diff={comparison['mean_diff'] * 100:.2f}%, p={comparison['p_value']:.4f}")
                        else:
                            eval_logger.debug(f"[Baseline] Skipping {task_name}: no valid scores found with score_key='{score_key}'")
                except Exception as e:
                    eval_logger.warning(f'[Baseline] Failed for {task_name}: {e}')
        return results
    else:
        return None
decontaminate_suffix = '_decontaminate'

def _run_generate_until_agentic(lm, requests: list[Instance], agentic_trace_mode: str='basic', response_cache: Optional[ResponseCache]=None) -> list[str]:
    responses: list[str] = []
    for req in requests:
        current_context, generation_kwargs, current_doc_to_visual, doc_to_text, doc_id, task_name, split = req.args
        if not callable(doc_to_text):
            raise ValueError('generate_until_agentic requires callable doc_to_text')
        max_agentic_steps = int(generation_kwargs.get('max_agentic_steps', 12))
        base_generation_kwargs = copy.deepcopy(generation_kwargs)
        base_generation_kwargs.pop('max_agentic_steps', None)
        model_outputs: list[str] = []
        previous_round_info = None
        final_response = ''
        full_round_trace: list[dict] = []
        for round_idx in range(max_agentic_steps):
            round_input_context = current_context
            if getattr(lm, 'is_simple', False):
                single_req = Instance(request_type='generate_until', arguments=(current_context, copy.deepcopy(base_generation_kwargs), current_doc_to_visual, doc_id, task_name, split), idx=0, metadata=req.metadata)
            else:
                current_doc = lm.task_dict[task_name][split][doc_id]

                def _agentic_doc_to_messages(_doc):
                    visuals = current_doc_to_visual(_doc)
                    if visuals is None:
                        visuals = []
                    content = []
                    for visual in visuals:
                        if isinstance(visual, dict):
                            content.append({'type': 'audio', 'url': visual})
                        elif isinstance(visual, str):
                            content.append({'type': 'video', 'url': visual})
                        else:
                            content.append({'type': 'image', 'url': visual})
                    content.append({'type': 'text', 'text': current_context})
                    return [{'role': 'user', 'content': content}]
                single_req = Instance(request_type='generate_until', arguments=(current_context, _agentic_doc_to_messages, copy.deepcopy(base_generation_kwargs), doc_id, task_name, split), idx=0, metadata=req.metadata)
            if response_cache is not None:
                current_raw_output = response_cache.execute(lm, 'generate_until', [single_req])[0]
            else:
                current_raw_output = lm.generate_until([single_req])[0]
            current_output, _ = unwrap_generation_output(current_raw_output)
            model_outputs.append(current_output)
            final_response = current_output
            step_payload = doc_to_text(lm.task_dict[task_name][split][doc_id], previous_output=model_outputs, round_idx=round_idx + 1, previous_round_info=previous_round_info)
            if isinstance(step_payload, tuple) and len(step_payload) == 5:
                visuals, next_context, terminal_signal, updated_outputs, next_round_info = step_payload
                if updated_outputs is not None:
                    model_outputs = list(updated_outputs)
                    if model_outputs:
                        final_response = model_outputs[-1]
                previous_round_info = next_round_info
                if agentic_trace_mode == 'full':
                    round_record = {'round_idx': round_idx + 1, 'round_input': round_input_context, 'model_output': current_output, 'terminal': bool(terminal_signal)}
                    if isinstance(next_round_info, dict):
                        round_record['state'] = next_round_info.get('state')
                        round_record['tool_result'] = next_round_info.get('last_tool_result')
                        round_record['tool_calls'] = next_round_info.get('tool_calls')
                        round_record['valid_tool_calls'] = next_round_info.get('valid_tool_calls')
                        round_record['invalid_steps'] = next_round_info.get('invalid_steps')
                    if next_context is not None:
                        round_record['next_input'] = next_context
                    full_round_trace.append(round_record)
                if terminal_signal:
                    break
                if next_context is not None:
                    current_context = next_context
                if visuals is not None:
                    current_doc_to_visual = lambda _doc, _visuals=visuals: _visuals
            elif isinstance(step_payload, str):
                current_context = step_payload
            else:
                break
        if previous_round_info is not None and (not (isinstance(final_response, str) and final_response.strip().startswith('{'))):
            state = previous_round_info.get('state', {}) if isinstance(previous_round_info, dict) else {}
            valid_tool_calls = float(previous_round_info.get('valid_tool_calls', previous_round_info.get('tool_calls', 0))) if isinstance(previous_round_info, dict) else 0.0
            invalid_steps = float(previous_round_info.get('invalid_steps', 0.0)) if isinstance(previous_round_info, dict) else 0.0
            fallback_payload = {'success': False, 'error': 'max_agentic_steps_reached', 'tool_calls': float(previous_round_info.get('tool_calls', 0)) if isinstance(previous_round_info, dict) else 0.0, 'valid_tool_calls': valid_tool_calls, 'invalid_steps': invalid_steps, 'state': state, 'last_model_output': final_response, 'trace': model_outputs}
            if isinstance(state, dict):
                for key in ['cash', 'days_elapsed', 'inventory', 'mobile_data_working']:
                    if key in state:
                        fallback_payload[key] = state[key]
            final_response = json.dumps(fallback_payload, ensure_ascii=False)
        if agentic_trace_mode == 'full':
            try:
                parsed_response = json.loads(final_response) if isinstance(final_response, str) else None
                if isinstance(parsed_response, dict):
                    parsed_response['agentic_trace_mode'] = 'full'
                    parsed_response['agentic_rounds'] = full_round_trace
                    final_response = json.dumps(parsed_response, ensure_ascii=False)
            except (TypeError, json.JSONDecodeError):
                pass
        responses.append(final_response)
    return responses

@positional_deprecated
def evaluate(lm, task_dict, limit: Optional[int]=None, offset: int=0, cache_requests: bool=False, rewrite_requests_cache: bool=False, bootstrap_iters: Optional[int]=100000, write_out: bool=False, log_samples: bool=True, system_instruction: Optional[str]=None, apply_chat_template: bool=False, fewshot_as_multiturn: bool=False, verbosity: str='INFO', distributed_executor_backend: str='accelerate', eval_server_launcher: Optional[Union[str, Callable]]=None, cli_args=None, response_cache: Optional[ResponseCache]=None):
    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)
    configs = collections.defaultdict(dict)
    samples = collections.defaultdict(list)
    requests = collections.defaultdict(list)
    results_agg = collections.defaultdict(dict)
    groups_agg = collections.defaultdict(dict)
    padding_requests = collections.defaultdict(int)
    task_hierarchy = collections.defaultdict(list)
    task_order = collections.defaultdict(int)
    task_group_alias = collections.defaultdict(dict)
    num_fewshot = collections.defaultdict(int)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    global_rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    eval_logger.info(f'Running on rank {global_rank} (local rank {local_rank})')

    def _infer_task_request_type(task_obj: Task) -> Optional[str]:
        if task_obj.instances:
            return task_obj.instances[0].request_type
        output_type = getattr(task_obj, 'OUTPUT_TYPE', None)
        if output_type == 'multiple_choice':
            return 'loglikelihood'
        if isinstance(output_type, str):
            return output_type
        return None
    eval_tasks = get_task_list(task_dict)
    name_to_task = {}
    if not log_samples:
        if not all(('bypass' not in getattr(task_output.task, '_metric_fn_list', {}).keys() for task_output in eval_tasks)):
            raise ValueError("log_samples must be True for 'bypass' metric-only tasks")
    if distributed_executor_backend == 'accelerate' and (not hasattr(lm, 'accelerator')):
        lm.accelerator = Accelerator()
    for task_output in eval_tasks:
        task = task_output.task
        task_name = task_output.task_name
        task.args = cli_args
        name_to_task[task_name] = task
        if type(task) == tuple:
            group_name, task = task
            task_hierarchy[group_name].append(task_name)
            versions[group_name] = 'N/A'
        else:
            group_name = None
            task_hierarchy[task_name] = []
        if task is None:
            continue
        versions[task_name] = task.VERSION
        configs[task_name] = dict(task.dump_config())
        if 'num_fewshot' in configs[task_name]:
            n_shot = configs[task_name]['num_fewshot']
        else:
            n_shot = 0
        num_fewshot[task_name] = n_shot
        if 'task_alias' in configs[task_name]:
            task_group_alias[task_name] = configs[task_name]['task_alias']
        if 'group_alias' in configs[task_name] and group_name not in task_group_alias and (group_name is not None):
            task_group_alias[group_name] = configs[task_name]['group_alias']
        limit = get_sample_size(task, limit)
        task.build_all_requests(limit=limit, offset=offset, rank=global_rank, world_size=world_size, cache_requests=cache_requests, rewrite_requests_cache=rewrite_requests_cache, system_instruction=system_instruction, apply_chat_template=apply_chat_template, fewshot_as_multiturn=fewshot_as_multiturn, chat_template=getattr(lm, 'apply_chat_template') if apply_chat_template else None, tokenizer_name=getattr(lm, 'tokenizer_name', '') if apply_chat_template else '')
        eval_logger.debug(f'Task: {task_output.task_name}; number of requests on this rank: {len(task._instances)}')
        if write_out:
            eval_logger.warning('DEPRECATION WARNING: --write_out is deprecated and will be removed in v0.5.0. Use --log_samples instead for saving model outputs and debugging. The write_out flag only prints the first few documents and impacts performance.')
            print_writeout(task)
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)
        if world_size > 1:
            if distributed_executor_backend == 'accelerate':
                instances_rnk = torch.tensor(len(task._instances), device=lm.device)
                gathered_item = lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            elif distributed_executor_backend == 'torchrun':
                instances_rnk = torch.tensor(len(task._instances), device=lm.device)
                gathered_item = torch.zeros(world_size * 1, dtype=instances_rnk.dtype, device=lm.device)
                dist.all_gather_into_tensor(gathered_item, instances_rnk)
                gathered_item = gathered_item.cpu().detach().numpy().tolist()
            else:
                raise ValueError(f"Invalid distributed_executor_backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'.")
            local_reqtype = _infer_task_request_type(task)
            reqtype = local_reqtype
            if dist.is_available() and dist.is_initialized():
                gathered_reqtypes = [None] * world_size
                dist.all_gather_object(gathered_reqtypes, local_reqtype)
                reqtype = next((rt for rt in gathered_reqtypes if rt is not None), None)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            if reqtype is None:
                eval_logger.warning(f'Task: {task_output.task_name}; unable to infer request type on rank {global_rank}, skipping padding computation.')
            else:
                padding_requests[reqtype] += numpad
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        local_reqtypes = list(requests.keys())
        gathered_reqtypes = [None] * world_size
        dist.all_gather_object(gathered_reqtypes, local_reqtypes)
        canonical_reqtypes = []
        for rank_reqtypes in gathered_reqtypes:
            if not rank_reqtypes:
                continue
            for reqtype in rank_reqtypes:
                if reqtype not in canonical_reqtypes:
                    canonical_reqtypes.append(reqtype)
    else:
        canonical_reqtypes = list(requests.keys())
    for reqtype in canonical_reqtypes:
        reqs = requests.get(reqtype, [])
        eval_logger.info('Running {} requests'.format(reqtype))
        cloned_reqs = []
        pad_source = reqs[-1] if reqs else None
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)
        if world_size > 1 and padding_requests[reqtype] > 0:
            if pad_source is None:
                eval_logger.warning(f'Running {reqtype} requests but could not find a pad source request on rank {global_rank}; skipping rank padding.')
            else:
                for _ in range(padding_requests[reqtype]):
                    cloned_reqs.extend([pad_source] * pad_source.repeats)
        if reqtype == 'generate_until_agentic':
            trace_mode = 'basic'
            if cli_args is not None:
                trace_mode = getattr(cli_args, 'agentic_trace_mode', 'basic')
            resps = _run_generate_until_agentic(lm, cloned_reqs, agentic_trace_mode=trace_mode, response_cache=response_cache)
        elif response_cache is not None:
            resps = response_cache.execute(lm, reqtype, cloned_reqs)
        else:
            resps = getattr(lm, reqtype)(cloned_reqs)
        for x, req in zip(resps, cloned_reqs):
            text, tc = unwrap_generation_output(x)
            req.resps.append(text)
            req.token_counts.append(tc)
        if is_budget_exceeded():
            eval_logger.warning("Token budget reached after '{}' requests. Skipping remaining request types.", reqtype)
            break
        if world_size > 1:
            if distributed_executor_backend == 'accelerate':
                lm.accelerator.wait_for_everyone()
            elif distributed_executor_backend == 'torchrun':
                dist.barrier()
            else:
                raise ValueError(f"Invalid distributed_executor_backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'.")
    lm.clean()
    RANK = global_rank
    WORLD_SIZE = world_size
    if eval_server_launcher is not None and RANK == 0:
        eval_server_launcher.launch()
    if world_size > 1:
        if distributed_executor_backend == 'accelerate':
            lm.accelerator.wait_for_everyone()
        elif distributed_executor_backend == 'torchrun':
            dist.barrier()
    for task_output in eval_tasks:
        task = task_output.task
        task.apply_filters()
        set_task_context(task_output.task_name)
        instances_by_doc_id = collections.defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        local_filter_keys = list(task.instances[0].filtered_resps.keys()) if task.instances else []
        if WORLD_SIZE > 1 and dist.is_available() and dist.is_initialized():
            gathered_filter_keys = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_filter_keys, local_filter_keys)
            filter_keys = []
            for rank_keys in gathered_filter_keys:
                if not rank_keys:
                    continue
                for filter_key in rank_keys:
                    if filter_key not in filter_keys:
                        filter_keys.append(filter_key)
        else:
            filter_keys = local_filter_keys
        if len(filter_keys) == 0:
            eval_logger.warning(f'Task: {task_output.task_name}; no filter keys available on rank {RANK}.')
            continue
        for filter_key in filter_keys:
            cli_reasoning_tags = getattr(cli_args, 'reasoning_tags', None) if cli_args else None
            task_reasoning_tags = getattr(task.config, 'reasoning_tags', None)
            reasoning_tags = parse_reasoning_tags_config(cli_value=cli_reasoning_tags, task_value=task_reasoning_tags)
            if cli_args is not None and (not cli_args.process_with_media):
                doc_iterator = create_iterator(enumerate(task.eval_docs_no_media), rank=RANK, limit=int(limit) if limit else None, world_size=WORLD_SIZE, offset=offset)
            else:
                doc_iterator = task.doc_iterator(rank=RANK, limit=limit, world_size=WORLD_SIZE, offset=offset)
            doc_iterator_for_counting = create_iterator(range(len(task.test_docs())), rank=RANK, limit=limit, world_size=WORLD_SIZE, offset=offset) if task.has_test_docs() else create_iterator(range(len(task.validation_docs())), rank=RANK, limit=limit, world_size=WORLD_SIZE, offset=offset)
            total_docs = sum((1 for _ in doc_iterator_for_counting))
            pbar = tqdm(total=total_docs, desc='Postprocessing', disable=RANK != 0)
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                if reasoning_tags is not None:
                    for req in requests:
                        raw_resp = req.filtered_resps[filter_key]
                        req.raw_filtered_resps[filter_key] = raw_resp
                        if isinstance(raw_resp, str):
                            req.filtered_resps[filter_key] = strip_reasoning_tags(raw_resp, reasoning_tags)
                        elif isinstance(raw_resp, list):
                            req.filtered_resps[filter_key] = [strip_reasoning_tags(r, reasoning_tags) if isinstance(r, str) else r for r in raw_resp]
                metrics = task.process_results(doc, [req.filtered_resps[filter_key] for req in requests])
                repeats = task.config.repeats if hasattr(task, 'config') and hasattr(task.config, 'repeats') else 1
                if repeats > 1 and len(requests) == repeats:
                    per_sample_scores = {}
                    for req in requests:
                        sample_metrics = task.process_results(doc, [req.filtered_resps[filter_key]])
                        for metric_name, value in sample_metrics.items():
                            if metric_name not in per_sample_scores:
                                per_sample_scores[metric_name] = []
                            per_sample_scores[metric_name].append(value)
                    for metric_name, scores in per_sample_scores.items():
                        task_output.per_sample_metrics[metric_name, filter_key].append(scores)
                if log_samples:
                    target = task.doc_to_target(doc)
                    saved_doc = {}
                    for key, value in doc.items():
                        if not is_multimodal_content(value):
                            saved_doc[key] = value
                    filtered_arguments = []
                    for req in requests:
                        for value in req.args:
                            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                filtered_arguments.append(value)
                    input_media = _collect_input_media(doc, filtered_arguments)
                    per_sample_tc = []
                    for req in requests:
                        if req.token_counts:
                            tc = req.token_counts[0]
                            per_sample_tc.append(tc.to_dict() if tc is not None else None)
                        else:
                            per_sample_tc.append(None)
                    example = {'doc_id': doc_id, 'doc': saved_doc, 'target': target, 'arguments': filtered_arguments, 'resps': [req.raw_filtered_resps.get(filter_key, req.resps) for req in requests], 'filtered_resps': [req.filtered_resps[filter_key] for req in requests], 'token_counts': per_sample_tc, 'doc_hash': hash_string(json.dumps(requests[0].doc, indent=2, default=handle_non_serializable, ensure_ascii=False))}
                    if input_media:
                        example['input_media'] = input_media
                    example.update(metrics)
                    task_output.logged_samples.append(example)
                for metric, value in metrics.items():
                    task_output.sample_metrics[metric, filter_key].append(value)
                pbar.update(1)
            pbar.close()
        set_task_context(None)
    if WORLD_SIZE > 1:
        for task_output in eval_tasks:
            if log_samples:
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                per_rank_samples = []
                for sample in task_output.logged_samples:
                    per_rank_samples.append(sample)
                torch.distributed.gather_object(obj=per_rank_samples, object_gather_list=full_samples, dst=0)
                if RANK == 0:
                    task_output.logged_samples = list(itertools.chain.from_iterable(full_samples))
            all_metric_keys = list(task_output.sample_metrics.keys())
            gathered_keys = [None] * WORLD_SIZE if RANK == 0 else None
            torch.distributed.gather_object(obj=all_metric_keys, object_gather_list=gathered_keys, dst=0)
            if RANK == 0:
                all_keys_set = set()
                for rank_keys in gathered_keys:
                    if rank_keys:
                        all_keys_set.update(rank_keys)
                canonical_keys = sorted(all_keys_set, key=lambda x: str(x))
            else:
                canonical_keys = None
            broadcast_list = [canonical_keys] if RANK == 0 else [None]
            torch.distributed.broadcast_object_list(broadcast_list, src=0)
            canonical_keys = broadcast_list[0]
            for metrics in canonical_keys:
                if metrics in task_output.sample_metrics:
                    pre_gather = task_output.sample_metrics[metrics]
                else:
                    pre_gather = []
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(obj=pre_gather, object_gather_list=metric_list, dst=0)
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(itertools.chain.from_iterable(metric_list))
            all_ps_keys = list(task_output.per_sample_metrics.keys())
            gathered_ps_keys = [None] * WORLD_SIZE if RANK == 0 else None
            torch.distributed.gather_object(obj=all_ps_keys, object_gather_list=gathered_ps_keys, dst=0)
            if RANK == 0:
                all_ps_set = set()
                for rank_keys in gathered_ps_keys:
                    if rank_keys:
                        all_ps_set.update(rank_keys)
                canonical_ps_keys = sorted(all_ps_set, key=lambda x: str(x))
            else:
                canonical_ps_keys = None
            broadcast_ps = [canonical_ps_keys] if RANK == 0 else [None]
            torch.distributed.broadcast_object_list(broadcast_ps, src=0)
            canonical_ps_keys = broadcast_ps[0]
            for metrics in canonical_ps_keys:
                if metrics in task_output.per_sample_metrics:
                    pre_gather = task_output.per_sample_metrics[metrics]
                else:
                    pre_gather = []
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(obj=pre_gather, object_gather_list=metric_list, dst=0)
                if RANK == 0:
                    task_output.per_sample_metrics[metrics] = list(itertools.chain.from_iterable(metric_list))
        dist.barrier()
    if RANK == 0:
        if eval_server_launcher is not None:
            eval_server_launcher.clean()
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
            task_output.calculate_clt_aggregate_metric()
            task_output.calculate_stability_metrics()
        results, samples, configs, versions, num_fewshot, higher_is_better = consolidate_results(eval_tasks)
        if bool(results):
            results, versions, show_group_table, *_ = consolidate_group_results(results, versions, task_dict)
        results_agg, group_agg = prepare_print_tasks(task_dict, results)
        subtask_list = get_subtask_list(task_dict)
        _higher_is_better = {}
        for group, task_list in subtask_list.items():
            if len(task_list) != 0:
                for task in task_list:
                    for m, h in higher_is_better[task].items():
                        if m not in _higher_is_better.keys():
                            _higher_is_better[m] = h
                        if m in _higher_is_better and _higher_is_better[m] is not None and (_higher_is_better[m] != h):
                            eval_logger.warning(f'Higher_is_better values for metric {m} in group {group} are not consistent. Defaulting to None.')
                            _higher_is_better[m] = None
                higher_is_better[group] = _higher_is_better
        results_dict = {'results': dict(results_agg.items()), **({'groups': dict(group_agg.items())} if bool(group_agg) & show_group_table else {}), 'group_subtasks': dict(reversed(subtask_list.items())), 'configs': dict(sorted(configs.items())), 'versions': dict(sorted(versions.items())), 'n-shot': dict(sorted(num_fewshot.items())), 'higher_is_better': dict(sorted(higher_is_better.items())), 'n-samples': {task_output.task_name: {'original': len(task_output.task.eval_docs), 'effective': min(limit if limit else len(task_output.task.eval_docs), len(task_output.task.eval_docs))} for task_output in eval_tasks}}
        if log_samples:
            results_dict['samples'] = dict(samples)
    else:
        results_dict = None
    if WORLD_SIZE > 1:
        if distributed_executor_backend == 'accelerate':
            Accelerator().wait_for_everyone()
        elif distributed_executor_backend == 'torchrun':
            dist.barrier()
        else:
            raise ValueError(f"Invalid distributed_executor_backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'.")
    return results_dict

def request_caching_arg_to_dict(cache_requests: str) -> dict:
    request_caching_args = {'cache_requests': cache_requests in {'true', 'refresh'}, 'rewrite_requests_cache': cache_requests == 'refresh', 'delete_requests_cache': cache_requests == 'delete'}
    return request_caching_args
