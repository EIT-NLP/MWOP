from lmms_eval.api.filter import Filter

class DecontaminationFilter(Filter):
    name = 'track_decontamination'

    def __init__(self, path) -> None:
        self._decontam_results = None

    def apply(self, resps, docs) -> None:
        pass
