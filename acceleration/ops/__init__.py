import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TRITON_DEBUG'] = '0'
from .cache import DynamicCacheSeqFirst
from .attention.base import DenseAttentionKernel
from .attention_threshold import BaseThreshold
from .mlp.base import DenseMLPKernel
__ATTENTION__ = {'base': DenseAttentionKernel}
__SPARSE_ATTENTION__ = {'base': DenseAttentionKernel}
__THRESHOLD__ = {'base': BaseThreshold}
__MLP__ = {'base': DenseMLPKernel}
__KV_CACHE__ = {'base': DynamicCacheSeqFirst}
__APPROXIMATOR__ = {}
