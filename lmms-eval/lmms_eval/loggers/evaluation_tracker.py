import json
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from datasets.utils.metadata import MetadataConfigs
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status
from lmms_eval.utils import eval_logger, get_file_datetime, get_file_task_name, get_results_filenames, get_sample_results_filenames, handle_non_serializable, hash_string, sanitize_list, sanitize_model_name, sanitize_task_name

@dataclass(init=False)
class GeneralConfigTracker:
    model_source: str = None
    model_name: str = None
    model_name_sanitized: str = None
    system_instruction: str = None
    system_instruction_sha: str = None
    fewshot_as_multiturn: bool = None
    chat_template: str = None
    chat_template_sha: str = None
    start_time: float = None
    end_time: float = None
    total_evaluation_time_seconds: str = None

    def __init__(self) -> None:
        self.start_time = time.perf_counter()

    @staticmethod
    def _get_model_name(model_args: str) -> str:

        def extract_model_name(model_args: str, key: str) -> str:
            args_after_key = model_args.split(key)[1]
            return args_after_key.split(',')[0]
        prefixes = ['peft=', 'delta=', 'pretrained=', 'model=', 'model_version=', 'model_name=', 'model_id=', 'path=', 'engine=']
        for prefix in prefixes:
            if prefix in model_args:
                return extract_model_name(model_args, prefix)
        return ''

    def log_experiment_args(self, model_source: str, model_args: str, system_instruction: str, chat_template: str, fewshot_as_multiturn: bool) -> None:
        self.model_source = model_source
        self.model_name = GeneralConfigTracker._get_model_name(model_args)
        self.model_name_sanitized = sanitize_model_name(self.model_name)
        self.system_instruction = system_instruction
        self.system_instruction_sha = hash_string(system_instruction) if system_instruction else None
        self.chat_template = chat_template
        self.chat_template_sha = hash_string(chat_template) if chat_template else None
        self.fewshot_as_multiturn = fewshot_as_multiturn

    def log_end_time(self) -> None:
        self.end_time = time.perf_counter()
        self.total_evaluation_time_seconds = str(self.end_time - self.start_time)

class EvaluationTracker:

    def __init__(self, output_path: str=None, hub_results_org: str='', hub_repo_name: str='', details_repo_name: str='', results_repo_name: str='', push_results_to_hub: bool=False, push_samples_to_hub: bool=False, public_repo: bool=False, token: str='', leaderboard_url: str='', point_of_contact: str='', gated: bool=False) -> None:
        self.general_config_tracker = GeneralConfigTracker()
        self.output_path = output_path
        self.push_results_to_hub = push_results_to_hub
        self.push_samples_to_hub = push_samples_to_hub
        self.public_repo = public_repo
        self.leaderboard_url = leaderboard_url
        self.point_of_contact = point_of_contact
        self.api = HfApi(token=token) if token else None
        self.gated_repo = gated
        if not self.api and (push_results_to_hub or push_samples_to_hub):
            raise ValueError("Hugging Face token is not defined, but 'push_results_to_hub' or 'push_samples_to_hub' is set to True. Please provide a valid Hugging Face token by setting the HF_TOKEN environment variable.")
        if self.api and hub_results_org == '' and (push_results_to_hub or push_samples_to_hub):
            hub_results_org = self.api.whoami()['name']
            eval_logger.warning(f"hub_results_org was not specified. Results will be pushed to '{hub_results_org}'.")
        if hub_repo_name == '':
            details_repo_name = details_repo_name if details_repo_name != '' else 'lmms-eval-results'
            results_repo_name = results_repo_name if results_repo_name != '' else details_repo_name
        else:
            details_repo_name = hub_repo_name
            results_repo_name = hub_repo_name
            eval_logger.warning('hub_repo_name was specified. Both details and results will be pushed to the same repository. Using hub_repo_name is no longer recommended, details_repo_name and results_repo_name should be used instead.')
        self.details_repo = f'{hub_results_org}/{details_repo_name}'
        self.details_repo_private = f'{hub_results_org}/{details_repo_name}-private'
        self.results_repo = f'{hub_results_org}/{results_repo_name}'
        self.results_repo_private = f'{hub_results_org}/{results_repo_name}-private'

    def save_results_aggregated(self, results: dict, samples: dict, datetime_str: str) -> None:
        self.general_config_tracker.log_end_time()
        if self.output_path:
            try:
                eval_logger.info('Saving results aggregated')
                task_hashes = {}
                if samples:
                    for task_name, task_samples in samples.items():
                        sample_hashes = [f'sample_{index}' + s['doc_hash'] for index, s in enumerate(task_samples)]
                        task_hashes[task_name] = hash_string(''.join(sample_hashes))
                results.update({'task_hashes': task_hashes})
                results.update(asdict(self.general_config_tracker))
                dumped = json.dumps(results, indent=2, default=handle_non_serializable, ensure_ascii=False)
                path = Path(self.output_path if self.output_path else Path.cwd())
                path = path.joinpath(self.general_config_tracker.model_name_sanitized)
                path.mkdir(parents=True, exist_ok=True)
                self.date_id = datetime_str.replace(':', '-')
                file_results_aggregated = path.joinpath(f'{self.date_id}_results.json')
                file_results_aggregated.open('w', encoding='utf-8').write(dumped)
                if self.api and self.push_results_to_hub:
                    repo_id = self.results_repo if self.public_repo else self.results_repo_private
                    self.api.create_repo(repo_id=repo_id, repo_type='dataset', private=not self.public_repo, exist_ok=True)
                    self.api.upload_file(repo_id=repo_id, path_or_fileobj=str(path.joinpath(f'{self.date_id}_results.json')), path_in_repo=os.path.join(self.general_config_tracker.model_name, f'{self.date_id}_results.json'), repo_type='dataset', commit_message=f'Adding aggregated results for {self.general_config_tracker.model_name}')
                    eval_logger.info(f'Successfully pushed aggregated results to the Hugging Face Hub. You can find them at: {repo_id}')
            except Exception as e:
                eval_logger.warning('Could not save results aggregated')
                eval_logger.info(repr(e))
        else:
            eval_logger.info('Output path not provided, skipping saving results aggregated')

    def save_results_samples(self, task_name: str, samples: dict) -> None:
        if self.output_path:
            try:
                path = Path(self.output_path if self.output_path else Path.cwd())
                path = path.joinpath(self.general_config_tracker.model_name_sanitized)
                path.mkdir(parents=True, exist_ok=True)
                file_results_samples = path.joinpath(f'{self.date_id}_samples_{task_name}.jsonl')
                for sample in samples:
                    sample['input'] = sample['arguments'][0]
                    sample['resps'] = sanitize_list(sample['resps'])
                    sample['filtered_resps'] = sanitize_list(sample['filtered_resps'])
                    if isinstance(sample['resps'], list) and len(sample['resps']) == 1:
                        sample['resps'] = sample['resps'][0]
                    if isinstance(sample['filtered_resps'], list) and len(sample['filtered_resps']) == 1:
                        sample['filtered_resps'] = sample['filtered_resps'][0]
                    if sample['resps'] == sample['filtered_resps']:
                        sample.pop('resps')
                    elif isinstance(sample['resps'], list) and len(sample['resps']) == 1 and (sample['resps'][0] == sample['filtered_resps']):
                        sample.pop('resps')
                    sample['target'] = str(sample['target'])
                    sample.pop('arguments')
                    sample.pop('doc')
                    sample_dump = json.dumps(sample, default=handle_non_serializable, ensure_ascii=False) + '\n'
                    with open(file_results_samples, 'a', encoding='utf-8') as f:
                        f.write(sample_dump)
                eval_logger.info(f'Saving samples to {file_results_samples}')
                if self.api and self.push_samples_to_hub:
                    repo_id = self.details_repo if self.public_repo else self.details_repo_private
                    self.api.create_repo(repo_id=repo_id, repo_type='dataset', private=not self.public_repo, exist_ok=True)
                    try:
                        if self.gated_repo:
                            headers = build_hf_headers()
                            r = get_session().put(url=f'https://huggingface.co/api/datasets/{repo_id}/settings', headers=headers, json={'gated': 'auto'})
                            hf_raise_for_status(r)
                    except Exception as e:
                        eval_logger.warning('Could not gate the repository')
                        eval_logger.info(repr(e))
                    self.api.upload_folder(repo_id=repo_id, folder_path=str(path), path_in_repo=self.general_config_tracker.model_name_sanitized, repo_type='dataset', commit_message=f'Adding samples results for {task_name} to {self.general_config_tracker.model_name}')
                    eval_logger.info(f'Successfully pushed sample results for task: {task_name} to the Hugging Face Hub. You can find them at: {repo_id}')
            except Exception as e:
                eval_logger.warning('Could not save sample results')
                eval_logger.info(repr(e))
        else:
            eval_logger.info('Output path not provided, skipping saving sample results')

    def recreate_metadata_card(self) -> None:
        eval_logger.info('Recreating metadata card')
        repo_id = self.details_repo if self.public_repo else self.details_repo_private
        files_in_repo = self.api.list_repo_files(repo_id=repo_id, repo_type='dataset')
        results_files = get_results_filenames(files_in_repo)
        sample_files = get_sample_results_filenames(files_in_repo)
        latest_task_results_datetime = defaultdict(lambda: datetime.min.isoformat())
        for file_path in sample_files:
            file_path = Path(file_path)
            filename = file_path.name
            model_name = file_path.parent
            task_name = get_file_task_name(filename)
            results_datetime = get_file_datetime(filename)
            task_name_sanitized = sanitize_task_name(task_name)
            samples_key = f'{model_name}__{task_name_sanitized}'
            results_key = f'{model_name}__results'
            latest_datetime = max(latest_task_results_datetime[samples_key], results_datetime)
            latest_task_results_datetime[samples_key] = latest_datetime
            latest_task_results_datetime[results_key] = max(latest_task_results_datetime[results_key], latest_datetime)
        card_metadata = MetadataConfigs()
        for file_path in results_files:
            file_path = Path(file_path)
            results_filename = file_path.name
            model_name = file_path.parent
            eval_date = get_file_datetime(results_filename)
            eval_date_sanitized = re.sub('[^\\w\\.]', '_', eval_date)
            results_filename = Path('**') / Path(results_filename).name
            config_name = f'{model_name}__results'
            sanitized_last_eval_date_results = re.sub('[^\\w\\.]', '_', latest_task_results_datetime[config_name])
            if eval_date_sanitized == sanitized_last_eval_date_results:
                current_results = card_metadata.get(config_name, {'data_files': []})
                current_results['data_files'].append({'split': eval_date_sanitized, 'path': [str(results_filename)]})
                card_metadata[config_name] = current_results
                card_metadata[config_name]['data_files'].append({'split': 'latest', 'path': [str(results_filename)]})
        for file_path in sample_files:
            file_path = Path(file_path)
            filename = file_path.name
            model_name = file_path.parent
            task_name = get_file_task_name(filename)
            eval_date = get_file_datetime(filename)
            task_name_sanitized = sanitize_task_name(task_name)
            eval_date_sanitized = re.sub('[^\\w\\.]', '_', eval_date)
            results_filename = Path('**') / Path(filename).name
            config_name = f'{model_name}__{task_name_sanitized}'
            sanitized_last_eval_date_results = re.sub('[^\\w\\.]', '_', latest_task_results_datetime[config_name])
            if eval_date_sanitized == sanitized_last_eval_date_results:
                current_details_for_task = card_metadata.get(config_name, {'data_files': []})
                current_details_for_task['data_files'].append({'split': eval_date_sanitized, 'path': [str(results_filename)]})
                card_metadata[config_name] = current_details_for_task
                card_metadata[config_name]['data_files'].append({'split': 'latest', 'path': [str(results_filename)]})
        latest_datetime = max(latest_task_results_datetime.values())
        latest_model_name = max(latest_task_results_datetime, key=lambda k: latest_task_results_datetime[k])
        last_results_file = [f for f in results_files if latest_datetime.replace(':', '-') in f][0]
        last_results_file_path = hf_hub_url(repo_id=repo_id, filename=last_results_file, repo_type='dataset')
        latest_results_file = load_dataset('json', data_files=last_results_file_path, split='train')
        results_dict = latest_results_file['results'][0]
        new_dictionary = {'all': results_dict}
        new_dictionary.update(results_dict)
        results_string = json.dumps(new_dictionary, indent=4)
        dataset_summary = 'Dataset automatically created during the evaluation run of model '
        if self.general_config_tracker.model_source == 'hf':
            dataset_summary += f'[{self.general_config_tracker.model_name}](https://huggingface.co/{self.general_config_tracker.model_name})\n'
        else:
            dataset_summary += f'{self.general_config_tracker.model_name}\n'
        dataset_summary += f'The dataset is composed of {len(card_metadata) - 1} configuration(s), each one corresponding to one of the evaluated task.\n\nThe dataset has been created from {len(results_files)} run(s). Each run can be found as a specific split in each configuration, the split being named using the timestamp of the run.The "train" split is always pointing to the latest results.\n\nAn additional configuration "results" store all the aggregated results of the run.\n\nTo load the details from a run, you can for instance do the following:\n'
        if self.general_config_tracker.model_source == 'hf':
            dataset_summary += f'```python\nfrom datasets import load_dataset\ndata = load_dataset(\n\t"{repo_id}",\n\tname="{latest_model_name}",\n\tsplit="latest"\n)\n```\n\n'
        dataset_summary += f"""## Latest results\n\nThese are the [latest results from run {latest_datetime}]({last_results_file_path.replace('/resolve/', '/blob/')}) (note that there might be results for other tasks in the repos if successive evals didn't cover the same tasks. You find each in the results and the "latest" split for each eval):\n\n```python\n{results_string}\n```"""
        card_data = DatasetCardData(dataset_summary=dataset_summary, repo_url=f'https://huggingface.co/{self.general_config_tracker.model_name}', pretty_name=f'Evaluation run of {self.general_config_tracker.model_name}', leaderboard_url=self.leaderboard_url, point_of_contact=self.point_of_contact)
        card_metadata.to_dataset_card_data(card_data)
        card = DatasetCard.from_template(card_data, pretty_name=card_data.pretty_name)
        card.push_to_hub(repo_id, repo_type='dataset')
