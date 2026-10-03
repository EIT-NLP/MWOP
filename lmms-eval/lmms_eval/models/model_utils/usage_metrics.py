import threading
from typing import Any, Dict, List, Optional
from loguru import logger as eval_logger
_USAGE_HISTORY: List[Dict[str, Any]] = []
_USAGE_LOCK = threading.Lock()
_BUDGET: Optional[int] = None
_BUDGET_EXCEEDED: bool = False
_CURRENT_TASK_CONTEXT: Optional[str] = None

def reset_usage_metrics() -> None:
    global _BUDGET, _BUDGET_EXCEEDED, _CURRENT_TASK_CONTEXT
    with _USAGE_LOCK:
        _USAGE_HISTORY.clear()
    _BUDGET = None
    _BUDGET_EXCEEDED = False
    _CURRENT_TASK_CONTEXT = None

def set_budget(max_tokens: Optional[int]=None) -> None:
    global _BUDGET
    _BUDGET = max_tokens

def set_task_context(task_name: Optional[str]) -> None:
    global _CURRENT_TASK_CONTEXT
    _CURRENT_TASK_CONTEXT = task_name

def log_usage(model_name: str, task_name: Optional[str]=None, input_tokens: int=0, output_tokens: int=0, reasoning_tokens: int=0, source: str='model') -> None:
    global _BUDGET_EXCEEDED
    if task_name is None:
        task_name = _CURRENT_TASK_CONTEXT
    record: Dict[str, Any] = {'model_name': model_name, 'task_name': task_name, 'input_tokens': input_tokens, 'output_tokens': output_tokens, 'reasoning_tokens': reasoning_tokens, 'source': source}
    with _USAGE_LOCK:
        _USAGE_HISTORY.append(record)
        _check_budget()
    eval_logger.debug('Usage: model={} task={} in={} out={} reason={} source={}', model_name, task_name, input_tokens, output_tokens, reasoning_tokens, source)

def is_budget_exceeded() -> bool:
    return _BUDGET_EXCEEDED

def get_running_totals() -> Dict[str, Any]:
    with _USAGE_LOCK:
        history = list(_USAGE_HISTORY)
    input_tokens = sum((r['input_tokens'] for r in history))
    output_tokens = sum((r['output_tokens'] for r in history))
    reasoning_tokens = sum((r['reasoning_tokens'] for r in history))
    total_tokens = input_tokens + output_tokens + reasoning_tokens
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'reasoning_tokens': reasoning_tokens, 'total_tokens': total_tokens, 'n_api_calls': len(history)}

def summarize_usage_metrics() -> Dict[str, Any]:
    with _USAGE_LOCK:
        history = list(_USAGE_HISTORY)
    if not history:
        return {}

    def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        inp = sum((r['input_tokens'] for r in records))
        out = sum((r['output_tokens'] for r in records))
        reason = sum((r['reasoning_tokens'] for r in records))
        return {'input_tokens': inp, 'output_tokens': out, 'reasoning_tokens': reason, 'total_tokens': inp + out + reason, 'n_api_calls': len(records)}
    total = _aggregate(history)
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for r in history:
        key = r['task_name'] if r['task_name'] is not None else '_unknown'
        by_task.setdefault(key, []).append(r)
    by_task_agg = {k: _aggregate(v) for k, v in by_task.items()}
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for r in history:
        by_source.setdefault(r['source'], []).append(r)
    by_source_agg = {k: _aggregate(v) for k, v in by_source.items()}
    return {'total': total, 'by_task': by_task_agg, 'by_source': by_source_agg, 'budget_exceeded': _BUDGET_EXCEEDED, 'budget_total_tokens': _BUDGET}

def _check_budget() -> None:
    global _BUDGET_EXCEEDED
    if _BUDGET is None or _BUDGET_EXCEEDED:
        return
    total = sum((r['input_tokens'] + r['output_tokens'] + r['reasoning_tokens'] for r in _USAGE_HISTORY))
    if total >= _BUDGET:
        _BUDGET_EXCEEDED = True
        eval_logger.warning('Token budget exceeded: {} / {} tokens used', f'{total:,}', f'{_BUDGET:,}')
