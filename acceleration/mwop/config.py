import hashlib
import json
from pathlib import Path
import torch

def load_plans(directory):
    directory = Path(directory)
    plans = [dict(flags=[[False] * 28 for _ in range(3)], keep=None) for _ in range(28)]
    hashes = {}
    names = ('V2V_40.json', 'T2V_60.json', 'T2T_10.json')

    def read(name):
        raw = (directory / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        obj = json.loads(raw)
        return obj
    for role, name in enumerate(names):
        obj = read(name)
        pairs = obj['heads']
        assert len(pairs) == len(set(map(tuple, pairs)))
        for layer, head in pairs:
            assert 0 <= layer < 28 and 0 <= head < 28
            plans[layer]['flags'][role][head] = True
    components = read('FFN_visual50_text0_L27first.json')['components']
    vision = next((c['neurons'] for c in components if c['token_scope'] == 'vision'))
    for c in components:
        assert c['token_scope'] in ('vision', 'text')
        if c['token_scope'] == 'text':
            assert not any(c['neurons'].values())
    assert all((str(int(k)) == k and 0 <= int(k) < 28 for k in vision))
    for layer, plan in enumerate(plans):
        indices = vision.get(str(layer), [])
        assert len(indices) == len(set(indices)) and all((0 <= x < 18944 for x in indices))
        pruned = set(indices)
        plan['keep'] = [x for x in range(18944) if x not in pruned]
        plan['layer'] = layer
    return (plans, hashes)

def flags_tensor(plan, device):
    return torch.tensor(plan['flags'], dtype=torch.bool, device=device).contiguous()
