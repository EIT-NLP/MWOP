from importlib import metadata
import torch

def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return 'missing'
print(f'python_torch={torch.__version__}')
print(f'torch_cuda={torch.version.cuda}')
print(f'cuda_available={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpu_name={torch.cuda.get_device_name(0)}')
    print(f'gpu_capability={torch.cuda.get_device_capability(0)}')
from llava.model import LlavaQwenForCausalLM
packages = ['torch', 'torchvision', 'torchaudio', 'transformers', 'tokenizers', 'accelerate', 'deepspeed', 'flash-attn', 'peft', 'bitsandbytes', 'llava', 'lmms-eval']
for name in packages:
    print(f'{name}=={version(name)}')
