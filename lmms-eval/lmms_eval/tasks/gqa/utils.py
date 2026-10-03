import os
from pathlib import Path
import datasets
import yaml
from datasets import load_dataset
from lmms_eval.dataset_paths import resolve_dataset_path
from tqdm import tqdm

def _load_gqa_task_config():
    with open(Path(__file__).parent / 'gqa.yaml', 'r', encoding='utf-8') as f:
        safe_lines = [line for line in f if '!function' not in line]
    return yaml.safe_load(''.join(safe_lines))

def _load_gqa_image_dataset():
    config = _load_gqa_task_config()
    dataset_path = resolve_dataset_path(config.get('dataset_path', 'lmms-lab/GQA'))
    dataset_kwargs = dict(config.get('dataset_kwargs') or {})
    resolved_dataset_path = os.path.expanduser(os.path.expandvars(dataset_path))
    if os.path.exists(resolved_dataset_path):
        dataset_path = resolved_dataset_path
        dataset_kwargs.pop('token', None)
    return load_dataset(dataset_path, 'testdev_balanced_images', split='testdev', **dataset_kwargs)

def gqa_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if 'image' in dataset.column_names:
        return dataset
    gqa_raw_image_dataset = _load_gqa_image_dataset()
    gqa_id2image = {}
    for row in tqdm(gqa_raw_image_dataset, desc='Loading GQA images'):
        gqa_id2image[row['id']] = row['image'].convert('RGB')

    def _process_doc(doc):
        image = gqa_id2image[doc['imageId']]
        return {'image': image}
    return dataset.map(_process_doc, num_proc=8)

def gqa_doc_to_visual(doc):
    image = [doc['image'].convert('RGB')]
    return image

def gqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc['question']
    pre_prompt = lmms_eval_specific_kwargs['pre_prompt']
    post_prompt = lmms_eval_specific_kwargs['post_prompt']
    return f'{pre_prompt}{question}{post_prompt}'
