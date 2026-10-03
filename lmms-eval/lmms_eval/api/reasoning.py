import re
from typing import List, Optional, Union

def strip_reasoning_tags(text: str, tag_pairs: List[List[str]]) -> str:
    result = text
    for start_tag, end_tag in tag_pairs:
        while start_tag in result and end_tag in result:
            start = result.find(start_tag)
            end = result.find(end_tag, start)
            if start != -1 and end != -1:
                result = result[:start] + result[end + len(end_tag):]
            else:
                break
        if end_tag in result and start_tag not in result:
            result = result.rsplit(end_tag, 1)[-1]
    return result.strip()

def parse_reasoning_tags_config(cli_value: Optional[str]=None, task_value: Optional[object]=None) -> Optional[List[List[str]]]:
    import json
    effective = task_value if task_value is not None else cli_value
    if effective is None or effective == 'none' or effective is False:
        return None
    if isinstance(effective, str):
        return json.loads(effective)
    return effective
