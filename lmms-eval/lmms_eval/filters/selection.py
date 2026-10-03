from collections import Counter
from lmms_eval.api.filter import Filter
from lmms_eval.api.registry import register_filter

@register_filter('take_first')
class TakeFirstFilter(Filter):

    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        return map(lambda r: r[0] if r else '', resps)

@register_filter('take_first_k')
class TakeKFilter(Filter):

    def __init__(self, **kwargs) -> None:
        self.k = kwargs.pop('k')
        super().__init__(**kwargs)

    def apply(self, resps, docs):
        resps = list(resps)
        assert len(resps[0]) >= self.k, f'Need at least {self.k} responses per doc to take first {self.k}, but got {len(resps[0])} only! Please increase TaskConfig.repeats .'
        selected = map(lambda r: r[:self.k], resps)
        return selected

@register_filter('majority_vote')
class MajorityVoteFilter(Filter):

    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):

        def select_majority(resp):
            counts = Counter(resp)
            vote = counts.most_common(1)[0][0]
            return vote
        return map(lambda r: [select_majority(r)], resps)
