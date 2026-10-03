import os
import pickle
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from loguru import logger
from torch.utils.collect_env import get_pretty_env_info
from transformers import __version__ as trans_version

def remove_none_pattern(input_string: str) -> Tuple[str, bool]:
    pattern = re.compile(',none$')
    result = re.sub(pattern, '', input_string)
    removed = result != input_string
    return (result, removed)

def is_serializable(o: Any) -> bool:
    try:
        pickle.dumps(o)
        return True
    except (pickle.PickleError, TypeError, AttributeError):
        return False

def _handle_non_serializable(o: Any) -> Union[int, str, list]:
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)

def get_commit_from_path(repo_path: Union[Path, str]) -> Optional[str]:
    try:
        git_folder = Path(repo_path, '.git')
        if git_folder.is_file():
            git_folder = Path(git_folder.parent, git_folder.read_text(encoding='utf-8').split('\n')[0].split(' ')[-1])
        if Path(git_folder, 'HEAD').exists():
            head_name = Path(git_folder, 'HEAD').read_text(encoding='utf-8').split('\n')[0].split(' ')[-1]
            head_ref = Path(git_folder, head_name)
            git_hash = head_ref.read_text(encoding='utf-8').replace('\n', '')
        else:
            git_hash = None
    except Exception as err:
        logger.debug(f'Failed to retrieve a Git commit hash from path: {str(repo_path)}. Error: {err}')
        return None
    return git_hash

def get_git_commit_hash():
    try:
        git_hash = subprocess.check_output(['git', 'describe', '--always']).strip()
        git_hash = git_hash.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = get_commit_from_path(os.getcwd())
    return git_hash

def add_env_info(storage: Dict[str, Any]):
    try:
        pretty_env_info = get_pretty_env_info()
    except Exception as err:
        pretty_env_info = str(err)
    transformers_version = trans_version
    upper_dir_commit = get_commit_from_path(Path(os.getcwd(), '..'))
    added_info = {'pretty_env_info': pretty_env_info, 'transformers_version': transformers_version, 'upper_git_hash': upper_dir_commit}
    storage.update(added_info)

def add_tokenizer_info(storage: Dict[str, Any], lm):
    if getattr(lm, 'tokenizer', False):
        try:
            tokenizer_info = {'tokenizer_pad_token': [lm.tokenizer.pad_token, str(lm.tokenizer.pad_token_id)], 'tokenizer_eos_token': [lm.tokenizer.eos_token, str(lm.tokenizer.eos_token_id)], 'tokenizer_bos_token': [lm.tokenizer.bos_token, str(lm.tokenizer.bos_token_id)], 'eot_token_id': getattr(lm, 'eot_token_id', None), 'max_length': getattr(lm, 'max_length', None)}
            storage.update(tokenizer_info)
        except Exception as err:
            logger.debug(f'Logging detailed tokenizer info failed with {err}, skipping...')
    else:
        logger.debug("LM does not have a 'tokenizer' attribute, not logging tokenizer metadata to results.")
