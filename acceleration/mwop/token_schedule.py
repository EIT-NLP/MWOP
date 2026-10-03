import math
import torch
RATIOS = ((7, 0.5436), (14, 0.2956), (21, 0.1607))

def layout(kind, prefix=18, visual=3215, suffix=19):
    assert kind in ('base', 'pdrop', 'zoo', 'upper')
    counts = []
    for layer in range(28):
        count = visual
        if kind == 'zoo':
            count = math.floor(visual * 0.5)
        if kind == 'pdrop':
            for boundary, ratio in RATIOS:
                if layer >= boundary:
                    count = math.floor(visual * ratio)
        counts.append(count)
    return [dict(layer=i, prefix=prefix, visual=v, suffix=suffix, length=prefix + v + suffix, span=[prefix, prefix + v]) for i, v in enumerate(counts)]

def override_plans(plans, kind):
    if kind != 'upper':
        return plans
    import copy
    result = copy.deepcopy(plans)
    for plan in result:
        plan['flags'][0] = [True] * 28
        plan['keep'] = []
    return result

def install(model, kind, prefix=18, visual=3215, suffix=19):
    rows = layout(kind, prefix, visual, suffix)
    model._mwop_layer_lengths = [r['length'] for r in rows]
    for r, layer in zip(rows, model.model.layers):
        region = layer.self_attn.mwop_region
        region.length = r['length']
        region.span = tuple(r['span'])
        region.block = None
        layer.mlp.span = tuple(r['span'])
    if kind in ('base', 'upper'):
        return rows
    device = model.model.embed_tokens.weight.device
    transitions = {}
    prev = visual
    original = list(range(prefix + visual + suffix))
    for row in rows:
        count = row['visual']
        if count != prev:
            idx = list(range(prefix)) + [prefix + j * prev // count for j in range(count)] + list(range(prefix + prev, prefix + prev + suffix))
            original = [original[i] for i in idx]
            transitions[row['layer']] = torch.tensor(idx, device=device, dtype=torch.long)
            row['selected_original_positions'] = list(original)
        prev = count
    state = {}

    def start(module, args, kwargs):
        state.clear()
        if kind == 'zoo':
            idx = transitions[0]
            kwargs = dict(kwargs)
            kwargs['inputs_embeds'] = kwargs['inputs_embeds'].index_select(1, idx)
            kwargs['attention_mask'] = kwargs['attention_mask'].index_select(1, idx)
            kwargs['position_ids'] = idx.unsqueeze(0)
        return (args, kwargs)
    model.model.register_forward_pre_hook(start, with_kwargs=True)
    for layer_id, layer in enumerate(model.model.layers):

        def before(module, args, kwargs, layer_id=layer_id):
            kwargs = dict(kwargs)
            if layer_id == 0:
                state.update(pe=kwargs['position_embeddings'], pos=kwargs['position_ids'], cache=kwargs['cache_position'])
            if layer_id in transitions and kind == 'pdrop':
                idx = transitions[layer_id]
                args = (args[0].index_select(1, idx), *args[1:])
                state['pe'] = tuple((t.index_select(1, idx) for t in state['pe']))
                state['pos'] = state['pos'].index_select(-1, idx)
                state['cache'] = state['cache'].index_select(0, idx)
            kwargs['position_embeddings'] = state['pe']
            kwargs['position_ids'] = state['pos']
            kwargs['cache_position'] = state['cache']
            return (args, kwargs)
        layer.register_forward_pre_hook(before, with_kwargs=True)
    return rows
