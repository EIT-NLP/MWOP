import torch
from benchmark_mwop import kv_tensors, write_json
from mwop.core_bench import error_stats

@torch.inference_mode()
def correctness(model, x, mask, output, backend):
    from mwop.attention import reference

    def compare(actual, expected):
        return dict(logits=error_stats(actual.logits, expected.logits, check=False), kv=[dict(layer=i, k=error_stats(ak, ek, check=False), v=error_stats(av, ev, check=False)) for i, ((ak, av), (ek, ev)) in enumerate(zip(kv_tensors(actual.past_key_values), kv_tensors(expected.past_key_values)))])

    def within(errors, calibration):
        if errors['logits']['relative_rms'] >= max(0.03, 3 * calibration['logits']['relative_rms']):
            return False
        return all((row[key]['relative_rms'] < max(0.05, 3 * calibration['kv'][i][key]['relative_rms']) for i, row in enumerate(errors['kv']) for key in ('k', 'v')))
    saved_flags = [layer.self_attn.mwop_region.flags.clone() for layer in model.model.layers]
    model.configure('dense')
    dense = model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
    try:
        for layer in model.model.layers:
            layer.self_attn.mwop_region.flags.zero_()
        model.configure('attention_only', backend='reference')
        dense_ref = model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
        baseline = compare(dense, dense_ref)
    finally:
        for layer, flags in zip(model.model.layers, saved_flags):
            layer.self_attn.mwop_region.flags.copy_(flags)
    write_json(output / 'dense_numerical_baseline.json', baseline)
    print('DENSE_NUMERICS', baseline['logits'], flush=True)
    del dense, dense_ref, saved_flags
    checks = []
    for mode in ('both',):
        model.configure(mode, backend='reference', reference_ffn=True)
        expected = model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
        model.configure(mode, backend=backend)
        local, handles = ([], [])

        def core_hook(i):

            def check(module, inputs, kwargs, result):
                ref = reference(kwargs['q'], kwargs['k'], kwargs['v'], module.flags, module.span)
                local.append(dict(layer=i, **error_stats(result, ref, check=False)))
            return check
        if mode in ('attention_only', 'both'):
            for i, layer in enumerate(model.model.layers):
                handles.append(layer.self_attn.mwop_region.register_forward_hook(core_hook(i), with_kwargs=True))
        try:
            actual = model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        row = dict(mode=mode, **compare(actual, expected), local_core=local, first_token_match=bool(torch.equal(actual.logits.argmax(-1), expected.logits.argmax(-1))))
        checks.append(row)
        write_json(output / 'model_correctness.json', checks)
        for err in local:
            assert err['relative_rms'] < 0.01 and err['max_row_relative_l2'] < 0.02, (mode, err)
        passed = within(row, baseline)
        if not passed and mode in ('attention_only', 'both') and (backend == 'triton'):
            model.configure(mode, backend='flex', reference_ffn=True)
            calibrated = model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
            row['same_mask_flex_baseline'] = compare(calibrated, expected)
            row['triton_flex'] = compare(actual, calibrated)
            passed = within(row, row['same_mask_flex_baseline']) and within(row['triton_flex'], baseline)
            del calibrated
        row['passed'] = passed
        write_json(output / 'model_correctness.json', checks)
        print('MODEL_CHECK', mode, row['logits'], 'same_mask_calibration', 'same_mask_flex_baseline' in row, flush=True)
        del expected, actual
        model.configure(mode, backend=backend)
        assert passed, (mode, row['logits'], 'See model_correctness.json for calibration and local checks')
    return checks
