from lmms_eval.api.filter import Filter

class LowercaseFilter(Filter):

    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):

        def filter_set(inst):
            return [resp.lower() for resp in inst]
        return [filter_set(resp) for resp in resps]

class UppercaseFilter(Filter):

    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):

        def filter_set(inst):
            return [resp.upper() for resp in inst]
        return [filter_set(resp) for resp in resps]

class MapFilter(Filter):

    def __init__(self, mapping_dict: dict={}, default_value=None) -> None:
        assert isinstance(mapping_dict, dict), 'Provided mapping_dict is not a dictionary'
        self.mapping_dict = mapping_dict
        self.default_value = default_value

    def apply(self, resps, docs):

        def filter_set(inst):
            return [self.mapping_dict.get(resp, self.default_value) for resp in inst]
        return [filter_set(resp) for resp in resps]
