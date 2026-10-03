import collections
import datetime
import fnmatch
import functools
import hashlib
import importlib.util
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import warnings
from typing import Any, Callable, Iterable, Iterator, List, Literal, Optional, Tuple, Union
import yaml
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')
import gc
from itertools import islice
import numpy as np
import pytz
import torch
import transformers
from jinja2 import BaseLoader, Environment, StrictUndefined
from loguru import logger as eval_logger
SPACING = ' ' * 47
HIGHER_IS_BETTER_SYMBOLS = {True: '↑', False: '↓'}

def is_json(string):
    try:
        json.loads(string)
        return True
    except json.JSONDecodeError:
        return False

def hash_string(string: str) -> str:
    return hashlib.sha256(string.encode('utf-8')).hexdigest()

def escaped_split(text, sep_char, maxsplit=-1):
    assert len(sep_char) == 1, 'separation string must be a single character for escaped splitting'
    if maxsplit == 0:
        return text
    maxsplit = max(0, maxsplit)
    return re.split('(?<!\\\\)' + sep_char, text, maxsplit)

def handle_arg_string(arg):
    if arg.lower() == 'true':
        return True
    elif arg.lower() == 'false':
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg

def handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)

def is_multimodal_content(value: Any) -> bool:
    if isinstance(value, (bytes, bytearray, np.ndarray, torch.Tensor)):
        return True
    if isinstance(value, dict):
        if 'array' in value or 'bytes' in value:
            return True
    try:
        from PIL import Image
        if isinstance(value, Image.Image):
            return True
    except ImportError:
        pass
    return False

def resolve_cache_dir(cache_dir: str, base_dir: Optional[str]=None) -> str:
    resolved = os.path.expanduser(os.path.expandvars(cache_dir))
    if base_dir is not None and (not os.path.isabs(resolved)):
        return os.path.join(base_dir, resolved)
    return resolved

def sanitize_list(sub):
    if isinstance(sub, list):
        return [sanitize_list(item) for item in sub]
    if isinstance(sub, tuple):
        return tuple((sanitize_list(item) for item in sub))
    else:
        return str(sub)

def _smart_comma_split(args_string):
    arg_list = []
    current_arg = []
    depth = 0
    in_quotes = False
    quote_char = None
    for i, char in enumerate(args_string):
        if char in ('"', "'") and (i == 0 or args_string[i - 1] != '\\'):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
        elif not in_quotes:
            if char in ('{', '['):
                depth += 1
            elif char in ('}', ']'):
                depth -= 1
            elif char == ',' and depth == 0:
                arg = ''.join(current_arg).strip()
                if arg:
                    arg_list.append(arg)
                current_arg = []
                continue
        current_arg.append(char)
    arg = ''.join(current_arg).strip()
    if arg:
        arg_list.append(arg)
    return arg_list

def simple_parse_args_string(args_string):
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = _smart_comma_split(args_string)
    args_dict = {k: handle_arg_string(v) for k, v in [arg.split('=', 1) for arg in arg_list]}
    return args_dict

def join_iters(iters):
    for iter in iters:
        yield from iter

def chunks(iter, n: int=0, fn=None):
    arr = []
    for i, x in enumerate(iter):
        arr.append(x)
        if len(arr) == (fn(i, iter) if fn else n):
            yield arr
            arr = []
    if arr:
        yield arr

def group(arr, fn):
    res = collections.defaultdict(list)
    for ob in arr:
        res[fn(ob)].append(ob)
    return list(res.values())

class MultiChoice:

    def __init__(self, choices) -> None:
        self.choices = choices

    def __contains__(self, values) -> bool:
        for value in values.split(','):
            if len(fnmatch.filter(self.choices, value)) == 0:
                eval_logger.info('Available tasks to choose:')
                for choice in self.choices:
                    eval_logger.info(f'  - {choice}')
                raise ValueError("'{}' is not in task list".format(value))
        return True

    def __iter__(self) -> Iterator:
        for choice in self.choices:
            yield choice

def pattern_match(patterns, source_list):
    if type(patterns) == str:
        patterns = [patterns]
    task_names = set()
    for pattern in patterns:
        try:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        except Exception as e:
            eval_logger.error(f'Error matching pattern {pattern}: {e}')
    return sorted(list(task_names))

def general_detokenize(string):
    string = string.replace(" n't", "n't")
    string = string.replace(' )', ')')
    string = string.replace('( ', '(')
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')
    string = re.sub(" (['.,])", '\\1', string)
    return string

def get_file_task_name(filename: str) -> str:
    return filename[filename.find('_') + 1:filename.rfind('_')]

def get_file_datetime(filename: str) -> str:
    return filename[filename.rfind('_') + 1:].replace('.jsonl', '')

def sanitize_model_name(model_name: str, full_path: bool=False) -> str:
    if full_path:
        return re.sub('[\\"<>:/\\|\\\\?\\*\\[\\]]+', '__', model_name)
    else:
        parts = model_name.split('/')
        last_two = '/'.join(parts[-2:]) if len(parts) > 1 else parts[-1]
        return re.sub('[\\"<>:/\\|\\\\?\\*\\[\\]]+', '__', last_two)

def sanitize_task_name(task_name: str) -> str:
    return re.sub('\\W', '_', task_name)

def get_latest_filename(filenames: List[str]) -> str:
    return max(filenames, key=lambda f: get_file_datetime(f))

def get_results_filenames(filenames: List[str]) -> List[str]:
    return [f for f in filenames if 'results' in f and '.json' in f]

def get_sample_results_filenames(filenames: List[str]) -> List[str]:
    return [f for f in filenames if '/samples_' in f and '.json' in f]

def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    pred_len = max_seq_len - context_len + 1
    predicted = 0
    first_seq_len = min(max_seq_len, len(token_list))
    yield ([prefix_token] + token_list[:first_seq_len - 1], token_list[:first_seq_len])
    predicted += first_seq_len
    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len
        yield (token_list[window_end - max_seq_len - 1:window_end - 1], token_list[window_end - window_pred_len:window_end])
        predicted += window_pred_len

def make_disjoint_window(pair):
    a, b = pair
    return (a[:len(a) - (len(b) - 1)], b)

class EnhancedJSONEncoder(json.JSONEncoder):

    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)

class Reorderer:

    def __init__(self, arr: List[Any], fn: Callable) -> None:
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        arr = [([y[0]], x[0][1]) for x in arr for y in x]
        arr.sort(key=lambda x: fn(x[1]))
        self.arr = arr

    def get_reordered(self):
        return [x[1] for x in self.arr]

    def get_original(self, newarr):
        res = [None] * self.size
        cov = [False] * self.size
        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds:
                res[ind] = v
                cov[ind] = True
        assert all(cov)
        return res

class Grouper:

    def __init__(self, arr, fn) -> None:
        self.size = len(arr)
        arr = list(enumerate(arr))

        def group_return_dict(arr, fn):
            res = collections.defaultdict(list)
            for ob in arr:
                res[fn(ob)].append(ob)
            return res
        arr = group_return_dict(arr, lambda x: fn(x[1]))
        self.arr = arr
        self._grouped = None

    def get_grouped(self):
        if self._grouped:
            return self._grouped
        grouped = {}
        for key in self.arr.keys():
            grouped[key] = [y[1] for y in self.arr[key]]
        self._grouped = grouped
        return grouped

    def get_original(self, grouped_dict):
        res = [None] * self.size
        cov = [False] * self.size
        assert grouped_dict.keys() == self.arr.keys()
        for key in grouped_dict.keys():
            for (ind, _), v in zip(self.arr[key], grouped_dict[key]):
                res[ind] = v
                cov[ind] = True
        assert all(cov)
        return res

def make_table(result_dict, column: str='results', sort_results: bool=False):
    from pytablewriter import LatexTableWriter, MarkdownTableWriter
    if column == 'results':
        column_name = 'Tasks'
    elif column == 'groups':
        column_name = 'Groups'
    all_headers = [column_name, 'Filter', 'n-shot', 'Metric', '', 'Value', '', 'Stderr', 'Stderr_CLT', 'Stderr_Clustered', 'EA', 'CA', 'IV', 'CR', 'Baseline', 'Diff', 'CI', 'P_Value']
    optional_col_indices = list(range(8, len(all_headers)))

    def fmt_se(se_val):
        if se_val is None or se_val == 'N/A':
            return 'N/A'
        if hasattr(se_val, '__len__') and len(se_val) == 0:
            return 'N/A'
        try:
            return '%.4f' % se_val
        except Exception:
            return 'N/A'
    values = []
    keys = result_dict[column].keys()
    if sort_results:
        keys = sorted(keys)
    for k in keys:
        dic = result_dict[column][k]
        n = str(result_dict.get('n-shot', ' ').get(k, ' '))
        higher_is_better = result_dict.get('higher_is_better', {}).get(k, {})
        if 'alias' in dic:
            k = dic.pop('alias')
        metric_items = dic.items()
        metric_items = sorted(metric_items)
        for mf, v in metric_items:
            m, _, f = mf.partition(',')
            if m.endswith('_stderr') or m.endswith('_stderr_clt') or m.endswith('_stderr_clustered'):
                continue
            if m.endswith('_expected_accuracy') or m.endswith('_consensus_accuracy'):
                continue
            if m.endswith('_internal_variance') or m.endswith('_consistency_rate'):
                continue
            if m.startswith('paired_'):
                continue
            hib = HIGHER_IS_BETTER_SYMBOLS.get(higher_is_better.get(m), '')
            v_numeric = v if isinstance(v, (int, float)) else None
            v = '%.4f' % v if isinstance(v, float) else v
            if v == '' or v is None:
                v = 'N/A'
            se = fmt_se(dic.get(m + '_stderr,' + f))
            se_clt = fmt_se(dic.get(m + '_stderr_clt,' + f))
            se_clustered = fmt_se(dic.get(m + '_stderr_clustered,' + f))
            ea = fmt_se(dic.get(m + '_expected_accuracy,' + f))
            ca = fmt_se(dic.get(m + '_consensus_accuracy,' + f))
            iv = fmt_se(dic.get(m + '_internal_variance,' + f))
            cr = fmt_se(dic.get(m + '_consistency_rate,' + f))
            baseline_name = dic.get('paired_baseline')
            baseline_str = str(baseline_name) if baseline_name else 'N/A'
            baseline_score = dic.get('paired_baseline_score')
            if v_numeric is not None and isinstance(baseline_score, (int, float)):
                diff = v_numeric - baseline_score
                diff_str = '%+.1f%%' % diff
            else:
                diff_str = 'N/A'
            ci_lower = dic.get('paired_ci_lower')
            ci_upper = dic.get('paired_ci_upper')
            if isinstance(ci_lower, (int, float)) and isinstance(ci_upper, (int, float)):
                ci_str = '[%+.1f%%, %+.1f%%]' % (ci_lower, ci_upper)
            else:
                ci_str = 'N/A'
            pval = dic.get('paired_pvalue')
            pval_str = '%.4f*' % pval if isinstance(pval, (int, float)) and pval < 0.05 else '%.4f' % pval if isinstance(pval, (int, float)) else 'N/A'
            is_empty = hasattr(v, '__len__') and (not isinstance(v, str)) and (len(v) == 0)
            if not is_empty:
                values.append([k, f, n, m, hib, v, '±', se, se_clt, se_clustered, ea, ca, iv, cr, baseline_str, diff_str, ci_str, pval_str])
    cols_to_hide = set()
    for col_idx in optional_col_indices:
        all_na = all((row[col_idx] == 'N/A' for row in values)) if values else True
        if all_na:
            cols_to_hide.add(col_idx)
    final_headers = [h for i, h in enumerate(all_headers) if i not in cols_to_hide]
    final_values = [[v for i, v in enumerate(row) if i not in cols_to_hide] for row in values]
    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = final_headers
    latex_writer.headers = final_headers
    md_writer.value_matrix = final_values
    latex_writer.value_matrix = final_values
    output_tables = [md_writer.dumps()]
    if column == 'results':
        throughput = result_dict.get('throughput', {})
        if isinstance(throughput, dict) and throughput:
            preferred_order = ['total_gen_tokens', 'total_elapsed_time', 'avg_latency', 'avg_speed']
            ordered_keys = preferred_order + sorted([k for k in throughput.keys() if k not in preferred_order])

            def get_unit(metric_name: str) -> str:
                if metric_name == 'total_gen_tokens':
                    return 'tokens'
                if metric_name == 'total_elapsed_time':
                    return 'seconds'
                if metric_name == 'avg_latency':
                    return 'seconds/request'
                if metric_name == 'avg_speed':
                    return 'tokens/s'
                return 'varies'
            throughput_summary = MarkdownTableWriter()
            throughput_summary.headers = ['Metric', 'Value', 'Unit']
            throughput_values = []
            for metric_name in ordered_keys:
                if metric_name not in throughput:
                    continue
                metric_value = throughput.get(metric_name)
                display_value = f'{metric_value:.4f}' if isinstance(metric_value, float) else str(metric_value)
                unit = get_unit(metric_name)
                throughput_values.append([metric_name, display_value, unit])
            throughput_summary.value_matrix = throughput_values
            output_tables.extend(['Throughput Summary', throughput_summary.dumps()])
    return '\n\n'.join(output_tables)

def positional_deprecated(fn):

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if len(args) != 1 if inspect.ismethod(fn) else 0:
            print(f'WARNING: using {fn.__name__} with positional arguments is deprecated and will be disallowed in a future version of lmms-evaluation-harness!')
        return fn(*args, **kwargs)
    return _wrapper

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

def get_git_commit_hash():
    try:
        git_hash = subprocess.check_output(['git', 'describe', '--always']).strip()
        git_hash = git_hash.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = None
    return git_hash

def get_git_branch_name():
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip()
        return branch.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def get_lmms_eval_version_string():
    branch = get_git_branch_name() or 'unknown'
    commit = get_git_commit_hash() or 'unknown'
    return f'{branch}@{commit[:8]}'

def get_lmms_eval_cache_version() -> str:
    commit = get_git_commit_hash()
    if commit:
        return commit
    try:
        import importlib.metadata
        return importlib.metadata.version('lmms-eval')
    except Exception:
        return 'unknown'
LMMS_EVAL_MOTTOS = ['We build trusted evaluation for probing real intelligence.', 'Better evals lead to better models.', 'Mapping the border of model capabilities.', 'Shaping what we build next, one benchmark at a time.', 'The unified evaluation toolkit for frontier models.', 'Probing abilities in the real world.', 'Good evaluation shapes what we build next.', 'Measure twice, train once.', 'Evaluation is the compass of progress.', 'Where rigorous benchmarks meet real-world intelligence.']

def get_eval_banner(branch: str=None, commit: str=None) -> str:
    import random
    motto = random.choice(LMMS_EVAL_MOTTOS)
    branch = branch or get_git_branch_name() or 'unknown'
    commit = commit or get_git_commit_hash() or 'unknown'
    lines = ['', 'LMMs-Eval: Probing Intelligence in the Real World', f'> {motto}', '', f'branch: {branch}', f'commit: {commit}', '']
    return '\n'.join(lines)

def get_datetime_str(timezone='Asia/Singapore'):
    tz = pytz.timezone(timezone)
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    local_time = utc_now.astimezone(tz)
    return local_time.strftime('%Y%m%d_%H%M%S')

def sanitize_long_string(s, max_length=40):
    if len(s) > max_length:
        return s[:max_length // 2] + '...' + s[-max_length // 2:]
    return s

def ignore_constructor(loader, node):
    return node

def import_function(loader, node):
    function_name = loader.construct_scalar(node)
    yaml_path = os.path.dirname(loader.name)
    *module_name, function_name = function_name.split('.')
    if isinstance(module_name, list):
        module_name = '.'.join(module_name)
    module_path = os.path.normpath(os.path.join(yaml_path, '{}.py'.format(module_name)))
    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, function_name)
        return function
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        return function
    except Exception as ex:
        raise ImportError(f"Failed to import function '{function_name}' from module '{module_name}'. Tried relative path '{module_path}' and absolute import.") from ex

def load_yaml_config(yaml_path=None, yaml_config=None, yaml_dir=None, mode='full'):
    if mode == 'simple':
        constructor_fn = ignore_constructor
    elif mode == 'full':
        constructor_fn = import_function
    yaml.add_constructor('!function', constructor_fn)
    if yaml_config is None:
        with open(yaml_path, 'rb') as file:
            yaml_config = yaml.full_load(file)
    if yaml_dir is None:
        yaml_dir = os.path.dirname(yaml_path)
    assert yaml_dir is not None
    assert yaml_config is not None
    if 'include' in yaml_config:
        include_path = yaml_config['include']
        del yaml_config['include']
        if isinstance(include_path, str):
            include_path = [include_path]
        include_path.reverse()
        final_yaml_config = {}
        for path in include_path:
            if not os.path.isfile(path):
                path = os.path.join(yaml_dir, path)
            try:
                included_yaml_config = load_yaml_config(yaml_path=path, mode=mode)
                final_yaml_config.update(included_yaml_config)
            except Exception as ex:
                raise ex
        final_yaml_config.update(yaml_config)
        return final_yaml_config
    return yaml_config

def regex_replace(string, pattern, repl, count: int=0):
    return re.sub(pattern, repl, string, count=count)
env = Environment(loader=BaseLoader, undefined=StrictUndefined)
env.filters['regex_replace'] = regex_replace

def apply_template(template: str, doc: dict) -> str:
    rtemplate = env.from_string(template)
    return rtemplate.render(**doc)

def create_iterator(raw_iterator, rank, world_size, limit=None, offset=0):
    if offset is None:
        offset = 0
    rank = int(rank)
    world_size = int(world_size)
    offset = int(offset)
    if offset < 0:
        raise ValueError(f'offset must be >= 0, got {offset}')
    if limit is not None:
        if isinstance(limit, float) and (not limit.is_integer()):
            raise ValueError(f'limit passed to create_iterator must be an integer count after normalization, got fractional value: {limit}')
        limit = int(limit)
        if limit < 0:
            raise ValueError(f'limit must be >= 0, got {limit}')
    start = rank + offset
    stop = None if limit is None else offset + limit
    return islice(raw_iterator, start, stop, world_size)

def pad_and_concat(max_length: int, tensors: List[torch.Tensor], padding_side: Literal['right', 'left']='right'):
    assert padding_side == 'left' or padding_side == 'right', f"Unrecognized padding type: '{padding_side}' not 'left' or 'right'"
    for i, tensor in enumerate(tensors):
        if len(tensor.shape) == 2:
            tensor = tensor.squeeze(0)
        tensor_len = tensor.shape[0]
        if tensor_len < max_length:
            if padding_side == 'right':
                tensors[i] = torch.cat([tensor, torch.zeros(max_length - tensor_len, dtype=torch.long, device=tensor.device)], dim=0).unsqueeze(0)
            else:
                tensors[i] = torch.cat([torch.zeros(max_length - tensor_len, dtype=torch.long, device=tensor.device), tensor], dim=0).unsqueeze(0)
        else:
            tensors[i] = tensor.unsqueeze(0)
    return torch.cat(tensors, dim=0)

def clear_torch_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()

def get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, str) and dtype != 'auto':
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype

class MultiTokenEOSCriteria(transformers.StoppingCriteria):

    def __init__(self, sequence: str, tokenizer: transformers.PreTrainedTokenizer, initial_decoder_input_length: int, batch_size: int) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length:][:, -self.sequence_id_len:]
        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker

def stop_sequences_criteria(tokenizer: transformers.PreTrainedTokenizer, stop_sequences: List[str], initial_decoder_input_length: int, batch_size: int) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList([*[MultiTokenEOSCriteria(sequence, tokenizer, initial_decoder_input_length, batch_size) for sequence in stop_sequences]])

def divide(iterable, n) -> List[Iterator]:
    if n < 1:
        raise ValueError('n must be at least 1')
    try:
        iterable[:0]
    except TypeError:
        seq = tuple(iterable)
    else:
        seq = iterable
    q, r = divmod(len(seq), n)
    ret = []
    stop = 0
    for i in range(1, n + 1):
        start = stop
        stop += q + 1 if i <= r else q
        ret.append(iter(seq[start:stop]))
    return ret

class Collator:

    def __init__(self, arr: List, sort_fn: Callable, group_fn: Callable=lambda x: x[1], grouping: bool=False) -> None:
        self.grouping = grouping
        self.fn = sort_fn
        self.group_fn = lambda x: group_fn(x[1])
        self.reorder_indices: List = []
        self.size = len(arr)
        self.arr_with_indices: Iterable[Any] = tuple(enumerate(arr))
        if self.grouping is True:
            self.group_by_index()

    def group_by_index(self) -> None:
        self.arr_with_indices = self.group(self.arr_with_indices, fn=self.group_fn, values=False)

    def get_batched(self, n: int=1, batch_fn: Optional[Callable]=None) -> Iterator:
        if self.grouping:
            for key, values in self.arr_with_indices.items():
                values = self._reorder(values)
                batch = self.get_chunks(values, n=n, fn=batch_fn)
                yield from batch
        else:
            values = self._reorder(self.arr_with_indices)
            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch

    def _reorder(self, arr: Union[List, Tuple[Tuple[int, Any], ...]]) -> List:
        arr = sorted(arr, key=lambda x: self.fn(x[1]))
        self.reorder_indices.extend([x[0] for x in arr])
        yield from [x[1] for x in arr]

    def get_original(self, newarr: List) -> List:
        res = [None] * self.size
        cov = [False] * self.size
        for ind, v in zip(self.reorder_indices, newarr):
            res[ind] = v
            cov[ind] = True
        assert all(cov)
        return res

    def __len__(self):
        return self.size

    @staticmethod
    def group(arr: Iterable, fn: Callable, values: bool=False) -> Iterable:
        res = collections.defaultdict(list)
        for ob in arr:
            try:
                hashable_dict = tuple(((key, tuple(value) if isinstance(value, collections.abc.Iterable) else value) for key, value in sorted(fn(ob).items())))
                res[hashable_dict].append(ob)
            except TypeError:
                res[fn(ob)].append(ob)
        if not values:
            return res
        return res.values()

    @staticmethod
    def get_chunks(_iter, n: int=0, fn=None):
        arr = []
        _iter = tuple(_iter)
        for i, x in enumerate(_iter):
            arr.append(x)
            if len(arr) == (fn(i, _iter) if fn else n):
                yield arr
                arr = []
        if arr:
            yield arr
