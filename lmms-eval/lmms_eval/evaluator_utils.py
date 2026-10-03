import collections
import inspect
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from lmms_eval.api.group import ConfigurableGroup
from lmms_eval.api.metrics import aggregate_subtask_metrics, clustered_stderr, consensus_accuracy, consistency_rate, expected_accuracy, internal_variance, paired_ttest, pooled_sample_stderr, stderr_for_metric
from lmms_eval.api.task import Task
from lmms_eval.utils import eval_logger, positional_deprecated

class TaskOutput:

    def __init__(self, task=None, task_name=None, task_config=None, version=None, group_name=None, n_shot=None, task_alias=None, group_alias=None, is_group=None):
        self.task = task
        self.task_config = task_config
        self.task_name = task_name
        self.group_name = group_name
        self.version = version
        self.n_shot = n_shot
        self.task_alias = task_alias
        self.group_alias = group_alias
        self.is_group = is_group
        self.logged_samples = []
        self.sample_len = None
        self.sample_metrics = collections.defaultdict(list)
        self.per_sample_metrics = collections.defaultdict(list)
        self.agg_metrics = collections.defaultdict(list)
        self.args = None

    @classmethod
    def from_taskdict(cls, task_name: str, task):
        if isinstance(task, tuple):
            group_name, task = task
        else:
            group_name = None
        if not task:
            is_group = True
            return cls(task=task, task_name=task_name, is_group=is_group, group_name=group_name)
        version = task.VERSION
        task_config = dict(task.dump_config())
        if (n_shot := task_config.get('num_fewshot')) == 0:
            meta_config = task_config.get('metadata', {})
            if isinstance(meta_config, dict):
                n_shot = meta_config.get('num_fewshot', 0)
            else:
                eval_logger.info(f'No metadata found in task config for {task_name}, using default n_shot=0')
                n_shot = 0
        task_alias = task_config.get('alias')
        group_alias = task_config.get('group_alias')
        return cls(task=task, task_name=task_name, task_config=task_config, group_name=group_name, version=version, n_shot=n_shot, task_alias=task_alias, group_alias=group_alias)

    def calculate_aggregate_metric(self, bootstrap_iters=100000) -> None:
        for (metric, filter_key), items in self.sample_metrics.items():
            if metric in self.task.aggregation():
                agg_fn = self.task.aggregation()[metric]
                metric_key = f'{metric},{filter_key}'
                if 'args' in inspect.signature(agg_fn).parameters:
                    self.agg_metrics[metric_key] = agg_fn(items, args=self.task.args)
                else:
                    self.agg_metrics[metric_key] = agg_fn(items)
                self.sample_len = len(items)
                if isinstance(bootstrap_iters, int):
                    stderr_fn = stderr_for_metric(metric=agg_fn, bootstrap_iters=min(bootstrap_iters, 100) if metric in ['bleu', 'chrf', 'ter'] else bootstrap_iters)
                    self.agg_metrics[f'{metric}_stderr,{filter_key}'] = stderr_fn(items) if stderr_fn and len(items) > 1 else 'N/A'
                else:
                    raise ValueError(f"Received bootstrap_iters '{bootstrap_iters}' but expected an integer. Set to 0 to turn off stderr calculations.")

    def calculate_clt_aggregate_metric(self) -> None:
        cluster_key = self.task_config.get('cluster_key') if self.task_config else None
        score_key = self.task_config.get('score_key', 'score') if self.task_config else 'score'
        for (metric, filter_key), items in self.sample_metrics.items():
            if metric not in self.task.aggregation():
                continue
            numeric_items = []
            cluster_ids = []
            for x in items:
                if isinstance(x, (int, float)):
                    numeric_items.append(x)
                    cluster_ids.append(None)
                elif isinstance(x, dict) and score_key in x:
                    numeric_items.append(x[score_key])
                    cluster_ids.append(x.get(cluster_key) if cluster_key else None)
            n = len(numeric_items)
            self.agg_metrics[f'{metric}_stderr_clt,{filter_key}'] = np.std(numeric_items, ddof=1) / np.sqrt(n) if n > 1 else 'N/A'
            valid_clusters = [c for c in cluster_ids if c is not None]
            if valid_clusters and len(set(valid_clusters)) > 1 and (n > 1):
                self.agg_metrics[f'{metric}_stderr_clustered,{filter_key}'] = clustered_stderr(numeric_items, cluster_ids)
            else:
                self.agg_metrics[f'{metric}_stderr_clustered,{filter_key}'] = 'N/A'

    def calculate_stability_metrics(self) -> None:
        repeats = self.task_config.get('repeats', 1) if self.task_config else 1
        if repeats <= 1:
            return
        score_key = self.task_config.get('score_key', 'score') if self.task_config else 'score'
        for (metric, filter_key), items in self.per_sample_metrics.items():
            if metric not in self.task.aggregation():
                continue
            scores_per_question = []
            for sample_scores in items:
                if not isinstance(sample_scores, list):
                    eval_logger.warning(f'Stability metrics: expected list of scores per question, got {type(sample_scores)}. Skipping.')
                    continue
                question_scores = []
                for x in sample_scores:
                    if isinstance(x, (int, float)):
                        question_scores.append(float(x))
                    elif isinstance(x, dict) and score_key in x:
                        question_scores.append(float(x[score_key]))
                    else:
                        eval_logger.debug(f'Stability metrics: cannot extract score from {type(x)}: {x}')
                if question_scores:
                    scores_per_question.append(question_scores)
            if not scores_per_question:
                eval_logger.warning(f'Stability metrics: no valid scores found for metric {metric}. Skipping.')
                continue
            self.agg_metrics[f'{metric}_expected_accuracy,{filter_key}'] = expected_accuracy(scores_per_question)
            self.agg_metrics[f'{metric}_consensus_accuracy,{filter_key}'] = consensus_accuracy(scores_per_question)
            self.agg_metrics[f'{metric}_internal_variance,{filter_key}'] = internal_variance(scores_per_question)
            self.agg_metrics[f'{metric}_consistency_rate,{filter_key}'] = consistency_rate(scores_per_question)

    def __repr__(self):
        return f'TaskOutput(task_name={self.task_name}, group_name={self.group_name}, version={self.version}, n_shot={self.n_shot}, task_alias={self.task_alias}, group_alias={self.group_alias})'

def get_task_list(task_dict: dict) -> List[TaskOutput]:
    outputs = []
    for task_name, task_obj in task_dict.items():
        if isinstance(task_obj, dict):
            _outputs = get_task_list(task_obj)
            outputs.extend(_outputs)
        else:
            task_output = TaskOutput.from_taskdict(task_name, task_obj)
            outputs.append(task_output)
    return outputs

def get_subtask_list(task_dict, task_root=None, depth=0):
    subtask_list = {}
    for group_obj, task_obj in task_dict.items():
        if isinstance(group_obj, ConfigurableGroup):
            group_name = group_obj.group_name
        else:
            group_name = group_obj
        if isinstance(task_obj, dict):
            _subtask_list = get_subtask_list(task_obj, task_root=group_name, depth=depth + 1)
            if task_root:
                subtask_list.setdefault((task_root, depth), []).extend([_task for _task, _depth in _subtask_list.keys() if _depth - 1 == depth])
            subtask_list = {**subtask_list, **_subtask_list}
        else:
            if isinstance(task_obj, ConfigurableGroup):
                group_or_task_name = task_obj.group_name
            elif isinstance(task_obj, Task):
                group_or_task_name = task_obj.task_name
            if task_root is None:
                subtask_list.setdefault((group_or_task_name, depth), [])
            else:
                subtask_list.setdefault((task_root, depth), []).append(group_or_task_name)
    if depth == 0:
        _subtask_list = {}
        for group_key, task_list in subtask_list.items():
            group_name, depth = group_key
            _subtask_list[group_name] = task_list
        subtask_list = _subtask_list
    return subtask_list

def print_writeout(task) -> None:
    for inst in task.instances:
        if inst.doc_id < 1:
            target = 'N/A (document is None)' if inst.doc is None else task.doc_to_target(inst.doc)
            eval_logger.info(f'Task: {task}; document {inst.doc_id}; context prompt (starting on next line):    \n{inst.args[0]}\n(end of prompt on previous line)\ntarget string or answer choice index (starting on next line):\n{target}\n(end of target on previous line)')
            eval_logger.info(f'Request: {str(inst)}')

def get_sample_size(task, limit: Optional[Union[int, float]]) -> Union[int, None]:
    if limit is None or limit == -1:
        return None
    if limit < 0:
        raise ValueError(f'limit must be -1 or non-negative, got {limit}')
    if 0 < limit < 1.0:
        return int(math.ceil(len(task.eval_docs) * limit))
    return int(limit)

def prepare_print_tasks(task_dict: dict, results: dict, task_depth=0, group_depth=0) -> Tuple[dict, dict]:

    def _sort_task_dict(task_dict):
        return dict(sorted(task_dict.items(), key=lambda item: item[0].group_name if isinstance(item[0], ConfigurableGroup) else item[0]))
    task_agg = collections.defaultdict(dict)
    group_agg = collections.defaultdict(dict)
    task_dict = _sort_task_dict(task_dict)
    for task_or_group_name, task_or_group_obj in task_dict.items():
        tab_string = ' ' * task_depth + '- ' if task_depth > 0 else ''
        if isinstance(task_or_group_name, ConfigurableGroup):
            name = task_or_group_name.group_name
            from_configurable_group = True
            task_or_group_obj = _sort_task_dict(task_or_group_obj)
        elif isinstance(task_or_group_name, str):
            name = task_or_group_name
            if isinstance(task_or_group_obj, Task):
                name = task_or_group_obj.task_name
            from_configurable_group = False
        task_agg[name] = results[name].copy()
        if from_configurable_group:
            if task_or_group_name.group_alias is not None:
                alias = task_or_group_name.group_alias
            else:
                alias = task_or_group_name.group
        elif 'alias' in task_agg[name]:
            alias = task_agg[name]['alias']
        else:
            alias = name
        task_agg[name]['alias'] = tab_string + alias
        if 'samples' in task_agg[name]:
            task_agg[name].pop('samples')
        if from_configurable_group and ' ' not in results[name]:
            group_tab_string = ' ' * group_depth + '- ' if group_depth > 0 else ''
            group_agg[name] = results[name].copy()
            group_agg[name]['alias'] = group_tab_string + alias
            if 'samples' in group_agg[name]:
                group_agg[name].pop('samples')
        if isinstance(task_or_group_obj, dict):
            task_depth += 1
            group_depth += 1
            _task_agg, _group_agg = prepare_print_tasks(task_or_group_obj, results, task_depth, group_depth)
            task_agg = {**task_agg, **_task_agg}
            group_agg = {**group_agg, **_group_agg}
            task_depth -= 1
            group_depth -= 1
    return (task_agg, group_agg)

def consolidate_results(eval_tasks: List[TaskOutput]) -> Tuple[dict, dict, dict, dict, dict, dict]:
    results = collections.defaultdict(dict)
    samples = collections.defaultdict(list)
    num_fewshot = collections.defaultdict(int)
    configs = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)
    higher_is_better = collections.defaultdict(dict)
    for task_output in eval_tasks:
        if 'task_alias' in (task_config := task_output.task_config):
            results[task_output.task_name]['alias'] = task_config['task_alias']
        else:
            results[task_output.task_name]['alias'] = task_output.task_name
        if (group_alias := task_output.group_alias):
            if group_alias not in results and (group_name := task_output.group_name):
                results[group_name]['alias'] = group_alias
        num_fewshot[task_output.task_name] = task_output.n_shot
        configs[task_output.task_name] = task_output.task_config
        versions[task_output.task_name] = task_output.version
        samples[task_output.task_name] = task_output.logged_samples
        higher_is_better[task_output.task_name] = task_output.task.higher_is_better()
        for (metric, filter_key), items in task_output.sample_metrics.items():
            metric_key = f'{metric},{filter_key}'
            results[task_output.task_name][metric_key] = task_output.agg_metrics[metric_key]
            results[task_output.task_name]['samples'] = task_output.sample_len
            results[task_output.task_name][f'{metric}_stderr,{filter_key}'] = task_output.agg_metrics[f'{metric}_stderr,{filter_key}']
            clt_key = f'{metric}_stderr_clt,{filter_key}'
            if clt_key in task_output.agg_metrics:
                results[task_output.task_name][clt_key] = task_output.agg_metrics[clt_key]
            clustered_key = f'{metric}_stderr_clustered,{filter_key}'
            if clustered_key in task_output.agg_metrics:
                results[task_output.task_name][clustered_key] = task_output.agg_metrics[clustered_key]
            for stat_suffix in ['expected_accuracy', 'consensus_accuracy', 'internal_variance', 'consistency_rate']:
                stat_key = f'{metric}_{stat_suffix},{filter_key}'
                if stat_key in task_output.agg_metrics:
                    results[task_output.task_name][stat_key] = task_output.agg_metrics[stat_key]
    return (results, samples, configs, versions, num_fewshot, higher_is_better)

def consolidate_group_results(results, versions, task_dict, task_root=None, show_group_table=False, task_aggregation_list=None) -> Tuple[dict, dict, bool, Union[None, dict]]:
    if task_root is None:
        task_root = {}
    if task_aggregation_list is None:
        task_aggregation_list = {}
    for group_or_task, group_or_task_info in task_dict.items():
        if isinstance(group_or_task, ConfigurableGroup):
            group_config = group_or_task.config
            group_or_task = group_or_task.group_name
        else:
            group_config = None
        if isinstance(group_or_task_info, Task):
            if task_root:
                task_aggregation_list.setdefault(task_root, []).append(group_or_task_info.task_name)
        else:
            results, versions, show_group_table, _task_aggregation_list = consolidate_group_results(results, versions, group_or_task_info, group_or_task, show_group_table, task_aggregation_list)
            if task_root:
                task_aggregation_list.setdefault(task_root, []).extend(task_aggregation_list.get(group_or_task, []))
            if group_config is None or group_config['aggregate_metric_list'] is None:
                results[group_or_task][' '] = ' '
                continue
            if 'aggregate_metric_list' in group_config:
                agg_metric_list = group_config['aggregate_metric_list']
            show_group_table = show_group_table | bool(group_config['aggregate_metric_list'])
            task_list = _task_aggregation_list[group_or_task]
            metric_list = list({key for task in task_list for key in results[task].keys() if '_stderr' not in key and key not in ['task', 'alias', 'samples']})
            for metric in metric_list:
                stderr = '_stderr,'.join(metric.split(','))
                metrics = [results[task][metric] for task in task_list if metric in results[task]]
                stderrs = [results[task][stderr] for task in task_list if stderr in results[task]]
                sizes = [results[task]['samples'] for task in task_list if metric in results[task]]
                for metric_config in agg_metric_list:
                    for filter_name in metric_config['filter_list']:
                        if metric_config['metric'] not in metric:
                            continue
                        if metric_config['aggregation'] == 'mean':
                            aggregate_fn = aggregate_subtask_metrics
                        elif callable(metric_config['aggregation']):
                            aggregate_fn = metric_config['aggregation']
                        else:
                            raise ValueError(f"Currently, only 'mean' is supported for automatically aggregating scores across groups' subtasks. Got '{metric_config['aggregation']}' for group '{group_or_task}'")
                        results[group_or_task][metric] = aggregate_fn(metrics, sizes, metric_config['weight_by_size'])
                        if 'N/A' in stderrs:
                            results[group_or_task][stderr] = 'N/A'
                        else:
                            results[group_or_task][stderr] = pooled_sample_stderr(stderrs, sizes)
                results[group_or_task]['samples'] = sum(sizes)
                group_metadata = group_config.get('metadata', None)
                if group_metadata is not None:
                    versions[group_or_task] = group_metadata.get('version', None)
    return (results, versions, show_group_table, task_aggregation_list)

@positional_deprecated
def find_test_root(start_path: pathlib.Path) -> pathlib.Path:
    cur_path = start_path.resolve()
    max_layers = 3
    for _ in range(max_layers):
        if (cur_path / 'tests' / 'test_version_stable.py').exists():
            return cur_path
        else:
            cur_path = cur_path.parent.resolve()
    raise FileNotFoundError(f'Unable to find package root within {max_layers} upwards' + f'of {start_path}')

@positional_deprecated
def run_task_tests(task_list: List[str]):
    import pytest
    package_root = find_test_root(start_path=pathlib.Path(__file__))
    task_string = ' or '.join(task_list)
    args = [f'{package_root}/tests/test_version_stable.py', f'--rootdir={package_root}', '-k', f'{task_string}']
    sys.path.append(str(package_root))
    pytest_return_val = pytest.main(args)
    if pytest_return_val:
        raise ValueError(f'Not all tests for the specified tasks ({task_list}) ran successfully! Error code: {pytest_return_val}')

def compute_baseline_comparison(current_scores: List[float], baseline_scores: List[float], baseline_name: str) -> Dict[str, Any]:
    result = paired_ttest(current_scores, baseline_scores)
    result['baseline_name'] = baseline_name
    result['baseline_mean'] = sum(baseline_scores) / len(baseline_scores) if baseline_scores else float('nan')
    result['current_mean'] = sum(current_scores) / len(current_scores) if current_scores else float('nan')
    return result
