from dataclasses import dataclass
from typing import List
from datasets import Dataset
from lmms_eval.api.instance import Instance

class Filter:

    def __init__(self, *args, **kwargs) -> None:
        pass

    def apply(self, resps, docs):
        return resps

@dataclass
class FilterEnsemble:
    name: str
    filters: List[Filter]

    def apply(self, instances: List[Instance], docs: List[Dataset]) -> None:
        resps = [inst.resps for inst in instances]
        for f in self.filters:
            resps = f.apply(resps, docs)
        for inst, resp in zip(instances, resps):
            inst.filtered_resps[self.name] = resp
