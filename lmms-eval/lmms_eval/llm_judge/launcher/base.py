from abc import ABC, abstractmethod

class BaseLauncher(ABC):

    def __init__(self, port: int=8000, host: str='localhost', timeout: int=1200, model: str=None, **kwargs):
        super().__init__()
        self.port = port
        self.host = host
        self.timeout = timeout
        if not model:
            raise ValueError('Specify the judge model explicitly when using a local launcher.')
        self.model = model

    @abstractmethod
    def launch(self, *args, **kwargs):
        pass

    @abstractmethod
    def clean():
        pass
