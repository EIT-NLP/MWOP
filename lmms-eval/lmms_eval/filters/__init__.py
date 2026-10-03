from lmms_eval.api.filter import Filter, FilterEnsemble
from . import extraction, selection, transformation
FILTER_REGISTRY = {'take_first': selection.TakeFirstFilter, 'regex': extraction.RegexFilter, 'majority_vote': selection.MajorityVoteFilter, 'take_first_k': selection.TakeKFilter, 'remove_whitespace': extraction.WhitespaceFilter, 'lowercase': transformation.LowercaseFilter, 'uppercase': transformation.UppercaseFilter, 'map': transformation.MapFilter, 'multi_choice_regex': extraction.MultiChoiceRegexFilter}

def get_filter(filter_name):
    if filter_name in FILTER_REGISTRY:
        return FILTER_REGISTRY[filter_name]
    else:
        return filter_name

def build_filter_ensemble(filter_name, components):
    filters = []
    for function, kwargs in components:
        if kwargs is None:
            f = get_filter(function)()
        else:
            f = get_filter(function)(**kwargs)
        filters.append(f)
    return FilterEnsemble(name=filter_name, filters=filters)
