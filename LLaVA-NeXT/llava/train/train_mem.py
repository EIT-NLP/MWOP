import torch as _torch
_orig_torch_load = _torch.load

def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
_torch.load = _torch_load_weights_only_false
from llava.train.train import train
if __name__ == '__main__':
    train()
