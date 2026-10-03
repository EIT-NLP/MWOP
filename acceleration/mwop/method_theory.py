def compute(config, method, model_config):
    p, v, s = [config[k] for k in ('prefix', 'visual', 'suffix')]
    d, i, hq, hk, dh = (model_config.hidden_size, model_config.intermediate_size, model_config.num_attention_heads, model_config.num_key_value_heads, 128)
    rows = []
    for layer in range(28):
        visual = v
        if method == 'zoo':
            visual = config['zoo_keep_visual']
        if method == 'pdrop':
            for boundary, count in zip(config['pdrop_boundaries'], config['pdrop_visual']):
                if layer >= boundary:
                    visual = count
        n = p + visual + s
        t = p + s
        qkvo = 2 * n * d * (2 * hq * dh + 2 * hk * dh)
        pairs = n * (n + 1) // 2
        width = i
        if method == 'shortv' and layer in config['shortv_layers']:
            qkvo = 2 * d * (2 * t * hq * dh + 2 * n * hk * dh)
            pairs = p * (p + 1) // 2 + s * (p + visual) + s * (s + 1) // 2
            width = 0
        if method == 'redundancy_lens':
            if layer in config['rl_attention_layers']:
                w = min(visual, config['rl_preceding_visual_window'] + 1)
                local = w * (w + 1) // 2 + (visual - w) * w
                pairs -= visual * (visual + 1) // 2 - local
            if layer in config['rl_ffn_layers']:
                width = config['rl_ffn_keep_channels']
        op = 4 * hq * dh * pairs
        ffn = 6 * d * (t * i + visual * width)
        rows.append(dict(layer=layer, length=n, visual_tokens=visual, vision_ffn_width=width, attention_op_TFLOP=op / 1000000000000.0, attention_module_TFLOP=(qkvo + op) / 1000000000000.0, ffn_module_TFLOP=ffn / 1000000000000.0, qkvo_TFLOP=qkvo / 1000000000000.0))
    totals = {key: sum((r[key] for r in rows)) for key in ('attention_op_TFLOP', 'attention_module_TFLOP', 'ffn_module_TFLOP', 'qkvo_TFLOP')}
    totals['last_token_lm_head_TFLOP'] = 2 * d * model_config.vocab_size / 1000000000000.0
    totals['model_TFLOP'] = totals['attention_module_TFLOP'] + totals['ffn_module_TFLOP'] + totals['last_token_lm_head_TFLOP']
    return dict(method=method, layers=rows, totals=totals, vocab_size=model_config.vocab_size, accounting='FMA=2; matrix multiplies only; causal QK/AV, QKVO, FFN and last-token LM head. No online scoring/sorting. Frozen indexed FFN width has no alignment padding.')
