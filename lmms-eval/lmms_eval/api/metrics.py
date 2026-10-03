import collections
import math
import random
import re
import string
from collections.abc import Iterable
from typing import Any, List
import numpy as np
import sacrebleu
from lmms_eval.api.registry import register_aggregation, register_metric

@register_aggregation('bypass')
def bypass_agg(arr):
    return 999

@register_aggregation('mean')
def mean(arr):
    return sum(arr) / len(arr)

@register_aggregation('median')
def median(arr):
    return arr[len(arr) // 2]

@register_aggregation('perplexity')
def perplexity(items):
    return math.exp(-mean(items))

@register_aggregation('weighted_perplexity')
def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))

@register_aggregation('bits_per_byte')
def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)

@register_aggregation('f1')
def f1_score(items):
    from sklearn.metrics import f1_score
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds)
    return np.max(fscore)

@register_aggregation('matthews_corrcoef')
def matthews_corrcoef(items):
    from sklearn.metrics import matthews_corrcoef
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return matthews_corrcoef(golds, preds)

@register_aggregation('bleu')
def bleu(items):
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score

@register_aggregation('chrf')
def chrf(items):
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score

@register_aggregation('ter')
def ter(items):
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_ter(preds, refs).score

@register_aggregation('brier_score')
def brier_score(items):
    gold, predictions = list(zip(*items))
    bs, num_class = np.array(predictions).shape
    gold = list(gold)
    gold_one_hot = np.eye(num_class)[gold]
    return np.mean(np.sum((predictions - gold_one_hot) ** 2, axis=1))

@register_metric(metric='brier_score', higher_is_better=False, output_type=['multiple_choice'], aggregation='brier_score')
def brier_score_fn(items):
    return items

@register_metric(metric='acc', higher_is_better=True, output_type=['loglikelihood', 'multiple_choice'], aggregation='mean')
def acc_fn(items):
    return items

@register_metric(metric='acc_norm', higher_is_better=True, output_type=['loglikelihood', 'multiple_choice'], aggregation='mean')
def acc_norm_fn(items):
    return items

@register_metric(metric='acc_mutual_info', higher_is_better=True, output_type='multiple_choice', aggregation='mean')
def acc_mutual_info_fn(items):
    return items

def exact_match_hf_evaluate(predictions, references, regexes_to_ignore=None, ignore_case=False, ignore_punctuation=False, ignore_numbers=False):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, '', x) for x in predictions])
            references = np.array([re.sub(s, '', x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)
    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)
    if ignore_punctuation:
        repl_table = string.punctuation.maketrans('', '', string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)
    if ignore_numbers:
        repl_table = string.digits.maketrans('', '', string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)
    score_list = predictions == references
    return {'exact_match': np.mean(score_list)}

@register_metric(metric='exact_match', higher_is_better=True, output_type='generate_until', aggregation='mean')
def exact_match_fn(**kwargs):
    return exact_match_hf_evaluate(**kwargs)

@register_metric(metric='perplexity', higher_is_better=False, output_type='loglikelihood', aggregation='perplexity')
def perplexity_fn(items):
    return items

@register_metric(metric='word_perplexity', higher_is_better=False, output_type='loglikelihood_rolling', aggregation='weighted_perplexity')
def word_perplexity_fn(items):
    return items

@register_metric(metric='byte_perplexity', higher_is_better=False, output_type='loglikelihood_rolling', aggregation='weighted_perplexity')
def byte_perplexity_fn(items):
    return items

@register_metric(metric='bits_per_byte', higher_is_better=False, output_type='loglikelihood_rolling', aggregation='bits_per_byte')
def bits_per_byte_fn(items):
    return items

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = (s2, s1)
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

@register_metric(metric='anls', higher_is_better=True, output_type='generate_until', aggregation='mean')
def anls(references, predictions, thresh_hold=0.5):
    values = []
    pred = predictions[0] if isinstance(predictions[0], str) else predictions[0][0]
    for answer in references:
        gt_answer = ' '.join(answer.strip().lower().split())
        det_answer = ' '.join(pred.strip().lower().split())
        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(pred.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))
    question_result = 1 - min(values)
    if question_result < thresh_hold:
        question_result = 0
    return {'anls': question_result}

def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))

def sample_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))

def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))

@register_metric(metric='bypass', higher_is_better=True, output_type=['loglikelihood', 'multiple_choice', 'generate_until', 'generate_until_multi_round'], aggregation='bypass')
def bypass(items):
    return items

@register_metric(metric='mcc', higher_is_better=True, output_type='multiple_choice', aggregation='matthews_corrcoef')
def mcc_fn(items):
    return items

@register_metric(metric='f1', higher_is_better=True, output_type='multiple_choice', aggregation='f1')
def f1_fn(items):
    return items

@register_metric(metric='bleu', higher_is_better=True, output_type=['generate_until', 'generate_until_multi_round'], aggregation='bleu')
def bleu_fn(items):
    return items

@register_metric(metric='chrf', higher_is_better=True, output_type=['generate_until', 'generate_until_multi_round'], aggregation='chrf')
def chrf_fn(items):
    return items

@register_metric(metric='ter', higher_is_better=True, output_type=['generate_until', 'generate_until_multi_round'], aggregation='ter')
def ter_fn(items):
    return items

@register_metric(metric='acc_all', higher_is_better=True, output_type='loglikelihood', aggregation='mean')
def acc_all(items):
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]
    for doc, pred in zip(docs, preds):
        paragraph_id = doc['idx']['paragraph']
        question_id = doc['idx']['question']
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[paragraph_id, question_id] = []
        gold_label = doc['label'] == 1
        question_scoring_dict[paragraph_id, question_id].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc

def acc_all_stderr(items):
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]
    for doc, pred in zip(docs, preds):
        question_id = doc['idx']['question']
        if question_id not in question_scoring_dict:
            question_scoring_dict[question_id] = []
        gold_label = doc['label'] == 1
        question_scoring_dict[question_id].append(gold_label == pred)
    acc = mean_stderr([int(all(x)) for x in question_scoring_dict.values()])
    return acc

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def weighted_mean(items):
    a, b = zip(*items)
    return sum(a) / sum(b)

def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and (not isinstance(obj, str))

def _sacreformat(refs, preds):
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f'Pred must be a str, was {preds[0]}'
        preds = [pred[0] for pred in preds]
    return (refs, preds)

class _bootstrap_internal:

    def __init__(self, f, n) -> None:
        self.f = f
        self.n = n

    def __call__(self, v):
        i, xs = v
        rnd = random.Random()
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))
        return res

def bootstrap_stderr(f, xs, iters):
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm
    print('bootstrapping for stddev:', f.__name__)
    for bootstrap in tqdm(pool.imap(_bootstrap_internal(f, chunk_size), [(i, xs) for i in range(iters // chunk_size)]), total=iters // chunk_size):
        res.extend(bootstrap)
    pool.close()
    return sample_stddev(res)

def bootstrap_chair_metric(metric_fn, xs, iters):
    print(f'bootstrapping for stddev: {metric_fn.__name__}')
    res = []
    from tqdm import tqdm
    for _ in tqdm(range(iters), desc='Bootstrap'):
        bootstrap_sample = random.choices(xs, k=len(xs))
        metric_value = metric_fn(bootstrap_sample)
        res.append(metric_value)
    return sample_stddev(res)

def stderr_for_metric(metric, bootstrap_iters: int):
    if bootstrap_iters <= 0:
        return None
    bootstrappable = [median, matthews_corrcoef, f1_score, perplexity, bleu, chrf, ter]
    try:
        from lmms_eval.tasks.amber_g.utils import amber_g_aggregate_chair, amber_g_aggregate_cog, amber_g_aggregate_cover, amber_g_aggregate_hal
        bootstrappable.extend([amber_g_aggregate_chair, amber_g_aggregate_cover, amber_g_aggregate_hal, amber_g_aggregate_cog])
    except Exception:
        pass
    try:
        from lmms_eval.tasks.coco_cap_chair.utils import coco_cap_chair_aggregate_results_chair_i, coco_cap_chair_aggregate_results_chair_s, coco_cap_chair_aggregate_results_recall
        bootstrappable.extend([coco_cap_chair_aggregate_results_chair_i, coco_cap_chair_aggregate_results_chair_s, coco_cap_chair_aggregate_results_recall])
    except Exception:
        pass
    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)
    if hasattr(metric, '__name__'):
        if 'coco_cap_chair' in metric.__name__:
            return lambda x: bootstrap_chair_metric(metric, x, iters=bootstrap_iters)
        if 'amber_g' in metric.__name__ or 'amber_' in metric.__name__:
            return lambda x: bootstrap_chair_metric(metric, x, iters=bootstrap_iters)
    stderr = {mean: mean_stderr, acc_all: acc_all_stderr}
    return stderr.get(metric, None)

def pooled_sample_stderr(stderrs: List[float], sizes: List[int]):
    assert len(stderrs) == len(sizes)
    pooled_sample_var = sum([(size - 1) * stderr ** 2 * size for size, stderr in zip(sizes, stderrs)]) / (sum(sizes) - len(sizes))
    return np.sqrt(pooled_sample_var / sum(sizes))

def combined_sample_stderr(stderrs: List[float], sizes: List[int], metrics=None):
    assert metrics is not None, "Need to pass a list of each subtask's metric for this stderr aggregation"
    assert len(stderrs) == len(sizes) and len(sizes) == len(metrics)
    variance = stderrs[0] ** 2
    curr_size = sizes[0]
    curr_score = metrics[0]
    for stderr, size, score in zip(stderrs[1:], sizes[1:], metrics[1:]):
        curr_score = (curr_score * curr_size + score * size) / (curr_size + size)
        variance = ((curr_size - 1) * variance + (size - 1) * stderr ** 2) / (curr_size + size - 1) + curr_size * size / ((curr_size + size) * (curr_size + size - 1)) * (curr_score - score) ** 2
    return np.sqrt(variance)

def aggregate_subtask_metrics(metrics, sizes, weight_by_size=True):
    if not weight_by_size:
        sizes = [1] * len(sizes)
    assert len(metrics) == len(sizes)
    return sum([metric * size for metric, size in zip(metrics, sizes)]) / sum(sizes)

def expected_accuracy(sample_scores: List[List[float]]) -> float:
    if not sample_scores:
        return float('nan')
    all_scores = [s for scores in sample_scores for s in scores]
    return sum(all_scores) / len(all_scores) if all_scores else float('nan')

def consensus_accuracy(sample_scores: List[List[float]]) -> float:
    if not sample_scores:
        return float('nan')
    correct = 0
    for scores in sample_scores:
        if not scores:
            continue
        if sum(scores) > len(scores) / 2:
            correct += 1
    return correct / len(sample_scores) if sample_scores else float('nan')

def internal_variance(sample_scores: List[List[float]]) -> float:
    if not sample_scores:
        return float('nan')
    variances = []
    for scores in sample_scores:
        if len(scores) < 2:
            continue
        mean_s = sum(scores) / len(scores)
        var = sum(((s - mean_s) ** 2 for s in scores)) / (len(scores) - 1)
        variances.append(var)
    return sum(variances) / len(variances) if variances else float('nan')

def consistency_rate(sample_scores: List[List[float]]) -> float:
    if not sample_scores:
        return float('nan')
    consistent = 0
    for scores in sample_scores:
        if not scores:
            continue
        if len(set(scores)) == 1:
            consistent += 1
    return consistent / len(sample_scores) if sample_scores else float('nan')

def clustered_stderr(scores: List[float], cluster_ids: List[Any]) -> float:
    n = len(scores)
    if n < 2:
        return float('nan')
    if len(scores) != len(cluster_ids):
        raise ValueError('scores and cluster_ids must have the same length')
    s_bar = sum(scores) / n
    var_scores = sum(((s - s_bar) ** 2 for s in scores)) / (n - 1)
    se_clt_squared = var_scores / n
    cluster_to_scores = collections.defaultdict(list)
    for i, (score, cid) in enumerate(zip(scores, cluster_ids)):
        cluster_to_scores[cid].append(score)
    cross_term = 0.0
    for cid, cluster_scores in cluster_to_scores.items():
        deviations = [s - s_bar for s in cluster_scores]
        cluster_sum = sum(deviations)
        sum_of_squares = sum((d * d for d in deviations))
        cross_term += cluster_sum * cluster_sum - sum_of_squares
    cross_term /= n * n
    return math.sqrt(se_clt_squared + cross_term)

def paired_ttest(current_scores: List[float], baseline_scores: List[float]) -> dict:
    from scipy import stats
    if len(current_scores) != len(baseline_scores):
        raise ValueError(f'Score lists must have same length: current={len(current_scores)}, baseline={len(baseline_scores)}')
    n = len(current_scores)
    if n < 2:
        return {'mean_diff': float('nan'), 'se_diff': float('nan'), 'ci_lower': float('nan'), 'ci_upper': float('nan'), 't_stat': float('nan'), 'p_value': float('nan'), 'n': n}
    diffs = [c - b for c, b in zip(current_scores, baseline_scores)]
    mean_diff = sum(diffs) / n
    var_diff = sum(((d - mean_diff) ** 2 for d in diffs)) / (n - 1)
    se_diff = math.sqrt(var_diff / n)
    if se_diff == 0:
        t_stat = float('inf') if mean_diff > 0 else float('-inf') if mean_diff < 0 else 0.0
        p_value = 0.0 if mean_diff != 0 else 1.0
    else:
        t_stat = mean_diff / se_diff
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    t_crit = stats.t.ppf(0.975, n - 1)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    return {'mean_diff': mean_diff, 'se_diff': se_diff, 'ci_lower': ci_lower, 'ci_upper': ci_upper, 't_stat': t_stat, 'p_value': p_value, 'n': n}

def power_analysis(effect_size: float, std_a: float=None, std_b: float=None, alpha: float=0.05, power: float=0.8, correlation: float=0.5, current_n: int=None) -> dict:
    from scipy import stats
    if std_a is None and std_b is None:
        std_a = std_b = 0.5
    elif std_a is not None and std_b is None:
        std_b = std_a
    elif std_a is None and std_b is not None:
        std_a = std_b
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    var_diff = std_a ** 2 + std_b ** 2 - 2 * correlation * std_a * std_b
    std_diff = math.sqrt(var_diff)
    d = effect_size / std_diff
    min_n = math.ceil(((z_alpha + z_beta) / d) ** 2)
    result = {'min_n': min_n, 'effect_size': effect_size, 'std_a': std_a, 'std_b': std_b, 'alpha': alpha, 'power': power, 'correlation': correlation}
    if current_n is not None:
        achieved_z_beta = d * math.sqrt(current_n) - z_alpha
        achieved_power = stats.norm.cdf(achieved_z_beta)
        result['current_n'] = current_n
        result['current_power'] = round(achieved_power, 4)
        mde = (z_alpha + z_beta) * std_diff / math.sqrt(current_n)
        result['min_detectable_effect'] = round(mde, 4)
    return result
