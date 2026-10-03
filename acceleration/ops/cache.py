import torch
from transformers import PretrainedConfig
from transformers.cache_utils import Cache, CacheLayerMixin
from typing import Optional, Any

class DynamicLayerSeqFirst(CacheLayerMixin):
    is_sliding = False

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = (key_states.dtype, key_states.device)
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]]=None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None:
            self.lazy_initialization(key_states)
        cur_len = self.get_seq_length()
        if cur_len > 0 and cache_kwargs.get('inplace_update_kvcache', False):
            self.keys[:, :key_states.shape[1]] = key_states
            self.values[:, :value_states.shape[1]] = value_states
        else:
            self.keys = torch.cat([self.keys, key_states], dim=1)
            self.values = torch.cat([self.values, value_states], dim=1)
        return (self.keys, self.values)

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return (kv_length, kv_offset)

    def get_seq_length(self) -> int:
        if self.keys is None or self.keys.numel() == 0:
            return 0
        return self.keys.shape[1]

    def get_max_cache_shape(self) -> int:
        return -1

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        self.keys = self.keys[:, :max_length].contiguous()
        self.values = self.values[:, :max_length].contiguous()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]

class DynamicCacheSeqFirst(Cache):

    def __init__(self, config: PretrainedConfig):
        layers = [DynamicLayerSeqFirst() for _ in range(config.num_hidden_layers)]
        if len(layers) == 0:
            super().__init__(layer_class_to_replicate=DynamicLayerSeqFirst)
        else:
            super().__init__(layers=layers)

    def fallback_cache(self, fallback_length: Optional[int]=1):
        for i in range(len(self.layers)):
            self.layers[i].crop(fallback_length)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        legacy_cache = ()
        for layer in self.layers:
            legacy_cache += ((layer.keys, layer.values),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: tuple[tuple[torch.Tensor, torch.Tensor]]) -> 'DynamicCacheSeqFirst':
        cache = cls()
        if past_key_values is None:
            logger.warning_once('past_key_values should not be None in from_legacy_cache()')
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
