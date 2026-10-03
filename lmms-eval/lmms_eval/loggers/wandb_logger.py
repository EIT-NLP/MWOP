import copy
import json
from typing import Any, Dict, List, Literal, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from packaging.version import Version
from lmms_eval.loggers.utils import _handle_non_serializable, remove_none_pattern

def get_wandb_printer() -> Literal['Printer']:
    from wandb.sdk.wandb_settings import Settings
    try:
        from wandb.sdk.lib.printer import get_printer
    except ImportError:
        from wandb.sdk.lib.printer import new_printer as get_printer
    printer = get_printer(Settings()._jupyter)
    return printer

class WandbLogger:

    def __init__(self, **kwargs) -> None:
        try:
            import wandb
            assert Version(wandb.__version__) >= Version('0.13.6')
            if Version(wandb.__version__) < Version('0.13.6'):
                wandb.require('report-editing:v0')
        except Exception as e:
            logger.warning(f'To use the wandb reporting functionality please install wandb>=0.13.6.\nTo install the latest version of wandb run `pip install wandb --upgrade`\n{e}')
        self.wandb_args: Dict[str, Any] = kwargs
        if wandb.run is None:
            self.run = wandb.init(**self.wandb_args)
        else:
            self.run = wandb.run
        self.printer = get_wandb_printer()

    def post_init(self, results: Dict[str, Any]) -> None:
        self.results: Dict[str, Any] = copy.deepcopy(results)
        self.task_names: List[str] = list(results.get('results', {}).keys())
        self.group_names: List[str] = list(results.get('groups', {}).keys())

    def _get_config(self) -> Dict[str, Any]:
        self.task_configs = self.results.get('configs', {})
        cli_configs = self.results.get('config', {})
        configs = {'task_configs': self.task_configs, 'cli_configs': cli_configs}
        return configs

    def _sanitize_results_dict(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        _results = copy.deepcopy(self.results.get('results', dict()))
        tmp_results = copy.deepcopy(_results)
        for task_name in self.task_names:
            task_result = tmp_results.get(task_name, dict())
            for metric_name, metric_value in task_result.items():
                _metric_name, removed = remove_none_pattern(metric_name)
                if removed:
                    _results[task_name][_metric_name] = metric_value
                    _results[task_name].pop(metric_name)
        wandb_summary = {}
        for task in self.task_names:
            task_result = _results.get(task, dict())
            for metric_name, metric_value in task_result.items():
                if isinstance(metric_value, str):
                    wandb_summary[f'{task}/{metric_name}'] = metric_value
        for summary_metric, summary_value in wandb_summary.items():
            _task, _summary_metric = summary_metric.split('/')
            _results[_task].pop(_summary_metric)
        tmp_results = copy.deepcopy(_results)
        for task_name, task_results in tmp_results.items():
            for metric_name, metric_value in task_results.items():
                _results[f'{task_name}/{metric_name}'] = metric_value
                _results[task_name].pop(metric_name)
        for task in self.task_names:
            _results.pop(task)
        return (wandb_summary, _results)

    def _log_results_as_table(self) -> None:
        columns = ['Version', 'Filter', 'num_fewshot', 'Metric', 'Value', 'Stderr']

        def make_table(columns: List[str], key: str='results'):
            import wandb
            table = wandb.Table(columns=columns)
            results = copy.deepcopy(self.results)
            for k, dic in results.get(key).items():
                if k in self.group_names and (not key == 'groups'):
                    continue
                version = results.get('versions').get(k)
                if version == 'N/A':
                    version = None
                n = results.get('n-shot').get(k)
                for mf, v in dic.items():
                    m, _, f = mf.partition(',')
                    if m.endswith('_stderr'):
                        continue
                    if m == 'alias':
                        continue
                    if m + '_stderr' + ',' + f in dic:
                        se = dic[m + '_stderr' + ',' + f]
                        if se != 'N/A':
                            se = '%.4f' % se
                        table.add_data(*[k, version, f, n, m, str(v), str(se)])
                    else:
                        table.add_data(*[k, version, f, n, m, str(v), ''])
            return table
        table = make_table(['Tasks'] + columns, 'results')
        self.run.log({'evaluation/eval_results': table})
        if 'groups' in self.results.keys():
            table = make_table(['Groups'] + columns, 'groups')
            self.run.log({'evaluation/group_eval_results': table})

    def _log_results_as_artifact(self) -> None:
        import wandb
        dumped = json.dumps(self.results, indent=2, default=_handle_non_serializable, ensure_ascii=False)
        artifact = wandb.Artifact('results', type='eval_results')
        with artifact.new_file('results.json', mode='w', encoding='utf-8') as f:
            f.write(dumped)
        self.run.log_artifact(artifact)

    def log_eval_result(self) -> None:
        configs = self._get_config()
        self.run.config.update(configs)
        wandb_summary, self.wandb_results = self._sanitize_results_dict()
        self.run.summary.update(wandb_summary)
        self.run.log(self.wandb_results)
        self._log_results_as_table()
        self._log_results_as_artifact()

    def _generate_dataset(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> pd.DataFrame:
        ids = [x['doc_id'] for x in data]
        labels = [x['target'] for x in data]
        instance = [''] * len(ids)
        resps = [''] * len(ids)
        filtered_resps = [''] * len(ids)
        model_outputs = {}
        metrics_list = config['metric_list']
        metrics = {}
        for metric in metrics_list:
            metric = metric.get('metric')
            if metric in ['word_perplexity', 'byte_perplexity', 'bits_per_byte']:
                metrics[f'{metric}_loglikelihood'] = [x[metric][0] for x in data]
                if metric in ['byte_perplexity', 'bits_per_byte']:
                    metrics[f'{metric}_bytes'] = [x[metric][1] for x in data]
                else:
                    metrics[f'{metric}_words'] = [x[metric][1] for x in data]
            else:
                metrics[metric] = [x[metric] for x in data]
        if config['output_type'] == 'loglikelihood':
            instance = [x['arguments'][0][0] for x in data]
            labels = [x['arguments'][0][1] for x in data]
            resps = [f"log probability of continuation is {x['resps'][0][0][0]} " + '\n\n' + 'continuation will {} generated with greedy sampling'.format('not be' if not x['resps'][0][0][1] else 'be') for x in data]
            filtered_resps = [f"log probability of continuation is {x['filtered_resps'][0][0]} " + '\n\n' + 'continuation will {} generated with greedy sampling'.format('not be' if not x['filtered_resps'][0][1] else 'be') for x in data]
        elif config['output_type'] == 'multiple_choice':
            instance = [x['arguments'][0][0] for x in data]
            choices = ['\n'.join([f'{idx}. {y[1]}' for idx, y in enumerate(x['arguments'])]) for x in data]
            resps = [np.argmax([n[0][0] for n in x['resps']]) for x in data]
            filtered_resps = [np.argmax([n[0] for n in x['filtered_resps']]) for x in data]
        elif config['output_type'] == 'loglikelihood_rolling':
            instance = [x['arguments'][0][0] for x in data]
            resps = [x['resps'][0][0] for x in data]
            filtered_resps = [x['filtered_resps'][0] for x in data]
        elif 'generate_until' in config['output_type']:
            instance = [x['arguments'][0][0] for x in data]
            resps = [x['resps'][0][0] for x in data]
            filtered_resps = [x['filtered_resps'][0] for x in data]
        model_outputs['raw_predictions'] = resps
        model_outputs['filtered_predictions'] = filtered_resps
        df_data = {'id': ids, 'data': instance}
        if config['output_type'] == 'multiple_choice':
            df_data['choices'] = choices
        tmp_data = {'input_len': [len(x) for x in instance], 'labels': labels, 'output_type': config['output_type']}
        df_data.update(tmp_data)
        df_data.update(model_outputs)
        df_data.update(metrics)
        return pd.DataFrame(df_data)

    def _log_samples_as_artifact(self, data: List[Dict[str, Any]], task_name: str) -> None:
        import wandb
        dumped = json.dumps(data, indent=2, default=_handle_non_serializable, ensure_ascii=False)
        artifact = wandb.Artifact(f'{task_name}', type='samples_by_task')
        with artifact.new_file(f'{task_name}_eval_samples.json', mode='w', encoding='utf-8') as f:
            f.write(dumped)
        self.run.log_artifact(artifact)

    def log_eval_samples(self, samples: Dict[str, List[Dict[str, Any]]]) -> None:
        task_names: List[str] = [x for x in self.task_names if x not in self.group_names]
        ungrouped_tasks = []
        tasks_by_groups = {}
        for task_name in task_names:
            group_names = self.task_configs[task_name].get('group', None)
            if group_names:
                if isinstance(group_names, str):
                    group_names = [group_names]
                for group_name in group_names:
                    if not tasks_by_groups.get(group_name):
                        tasks_by_groups[group_name] = [task_name]
                    else:
                        tasks_by_groups[group_name].append(task_name)
            else:
                ungrouped_tasks.append(task_name)
        for task_name in ungrouped_tasks:
            eval_preds = samples[task_name]
            df = self._generate_dataset(eval_preds, self.task_configs.get(task_name))
            self.run.log({f'{task_name}_eval_results': df})
            self._log_samples_as_artifact(eval_preds, task_name)
        for group, grouped_tasks in tasks_by_groups.items():
            grouped_df = pd.DataFrame()
            for task_name in grouped_tasks:
                eval_preds = samples[task_name]
                df = self._generate_dataset(eval_preds, self.task_configs.get(task_name))
                df['group'] = group
                df['task'] = task_name
                grouped_df = pd.concat([grouped_df, df], ignore_index=True)
                self._log_samples_as_artifact(eval_preds, task_name)
            self.run.log({f'{group}_eval_results': grouped_df})
