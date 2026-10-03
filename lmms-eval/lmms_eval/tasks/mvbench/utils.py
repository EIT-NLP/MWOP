import os
import re
import string
from pathlib import Path
import PIL
import yaml
DATA_LIST = {'object_interaction': 'star/Charades_segment', 'action_sequence': 'star/Charades_segment', 'action_prediction': 'star/Charades_segment', 'action_localization': 'sta/sta_video_segment', 'moving_count': 'clevrer/video_validation', 'fine_grained_pose': 'nturgbd_convert', 'character_order': 'perception/videos', 'object_shuffle': 'perception/videos', 'egocentric_navigation': 'vlnqa', 'moving_direction': 'clevrer/video_validation', 'episodic_reasoning': 'tvqa/video_fps3_hq_segment', 'fine_grained_action': 'Moments_in_Time_Raw/videos', 'scene_transition': 'scene_qa/video', 'state_change': 'perception/videos', 'moving_attribute': 'clevrer/video_validation', 'action_antonym': 'ssv2_video_mp4', 'unexpected_action': 'FunQA_test/test', 'counterfactual_inference': 'clevrer/video_validation', 'object_existence': 'clevrer/video_validation', 'action_count': 'perception/videos'}
hf_home = os.getenv('HF_HOME', '~/.cache/huggingface')
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / '_default_template_yaml', 'r') as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if '!function' not in line:
            safe_data.append(line)
config = yaml.safe_load(''.join(safe_data))
cache_name = config['dataset_kwargs']['cache_dir']

def _resolve_mvbench_cache_dir():
    cache_dir = os.path.expanduser(os.path.expandvars(cache_name))
    dataset_path = os.path.expanduser(os.path.expandvars(config.get('dataset_path', '')))
    candidates = []
    if os.path.isabs(cache_dir):
        candidates.append(cache_dir)
    candidates.append(os.path.join(base_cache_dir, cache_dir))
    if dataset_path and os.path.exists(dataset_path):
        candidates.extend([os.path.join(dataset_path, cache_dir), dataset_path, os.path.join(os.path.dirname(dataset_path), cache_dir)])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]

def _resolve_mvbench_media_path(cache_dir, dataset_folder, video):
    candidates = [os.path.join(cache_dir, dataset_folder, video), os.path.join(cache_dir, 'data0613', dataset_folder, video)]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"MVBench media does not exist; checked: {', '.join(candidates)}")

def mvbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    cache_dir = _resolve_mvbench_cache_dir()
    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs['sub_task']]
    video_path = _resolve_mvbench_media_path(cache_dir, dataset_folder, doc['video'])
    return [video_path]

def mvbench_frames_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    cache_dir = _resolve_mvbench_cache_dir()
    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs['sub_task']]
    video_path = _resolve_mvbench_media_path(cache_dir, dataset_folder, doc['video'])
    frame_path_list = sorted((os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg') or f.endswith('.png')))
    frame_image_list = [PIL.Image.open(frame_path).convert('RGB') for frame_path in frame_path_list]
    return frame_image_list

def mvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = ''
    option_list = doc['candidates']
    option_letters = string.ascii_uppercase
    for char_index, option in enumerate(option_list):
        option_letter = option_letters[char_index]
        option_prompt += f'({option_letter}) {option}\n'
    full_text = 'Question:' + doc['question'] + '\nOption:\n' + option_prompt + lmms_eval_specific_kwargs['post_prompt']
    return full_text

def mcq_acc(answer, pred):
    periodStrip = re.compile('(?!<=\\d)(\\.)(?!\\d)')
    commaStrip = re.compile('(\\d)(\\,)(\\d)')
    punct = [';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + ' ' in inText or ' ' + p in inText) or re.search(commaStrip, inText) != None:
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = periodStrip.sub('', outText, re.UNICODE)
        return outText

    def process(answer):
        option_regex = re.compile('^([A-E])\\.\\s*(.+)$', re.IGNORECASE)
        match = option_regex.match(answer.strip())
        if match:
            return match.group(1).upper()
        else:
            answer = answer.replace('\n', ' ')
            answer = answer.replace('\t', ' ')
            answer = answer.strip()
            answer = processPunctuation(answer)
            answer = answer.strip("'")
            answer = answer.strip('"')
            answer = answer.strip(')')
            answer = answer.strip('(')
            answer = answer.strip().lower()
            letter_match = re.search('\\b([A-E])\\b', answer, re.IGNORECASE)
            if letter_match:
                return letter_match.group(1).upper()
            return answer
    pred = process(pred)
    answer = process(answer)
    if pred == answer:
        score = 1
    else:
        score = 0
    return score

def mvbench_process_results(doc, results):
    pred = results[0]
    option_letters = string.ascii_uppercase
    gt_option_letter = None
    for i, candidate in enumerate(doc['candidates']):
        if candidate == doc['answer']:
            gt_option_letter = option_letters[i]
            break
    score = mcq_acc(gt_option_letter, pred)
    data_dict = {'pred_answer': pred, 'gt_answer': gt_option_letter, 'score': score}
    return {'mvbench_accuracy': data_dict}

def mvbench_aggregate_results(results):
    total_answered = 0
    total_correct = 0
    for result in results:
        if result['pred_answer'] != '':
            total_answered += 1
            total_correct += result['score']
    return 100 * total_correct / total_answered if total_answered > 0 else 0
