# several of the evaluation metrics are from https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/evaluation/metrics.py
"""Defines different metrics used for evaluation of tasks."""
import numpy as np
import math
import scipy
import sklearn
import collections
import sklearn.metrics
from logging import getLogger

import string	
import regex as re

from utils import string_to_float

logger = getLogger(__name__)


TASK_VAL_REFERENCES = {}

TASK_TEST_REFERENCES = {}

def average_multi_task(res_dict):
  return np.mean(sum([[vv for kk, vv in v.items()] for k, v in res_dict.items()], []))


def multi_task_gain(res_dict, ref_dict):
  res = []
  for task in res_dict:
    temp = []
    if task not in ref_dict:
      metrics = sum([list(res_dict[task].values()) for task in res_dict], [])
      return np.mean(metrics)
      
    for metric in res_dict[task]:
      temp.append((res_dict[task][metric] - ref_dict[task][metric]) / ref_dict[task][metric])

    res.append(np.mean(temp))
  return np.mean(res)


def accuracy(predictions, targets) -> dict:
    """Computes the average accuracy."""
    return {"accuracy": 100 * ((np.array(predictions) == np.array(targets)).mean())}

def pearson_corrcoef(predictions, targets) -> dict:
    """Computes Pearson correlation coefficient."""
    targets = [string_to_float(target) for target in targets]
    predictions= [string_to_float(prediction) for prediction in predictions]
    pearson_corrcoef = 100 * scipy.stats.pearsonr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(pearson_corrcoef):
        pearson_corrcoef = 0
    return {"pearson": pearson_corrcoef}

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def calculate_rouge(output_lns, reference_lns, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

def spearman_corrcoef(predictions, targets) -> dict:
    """Computes Spearman correlation coefficient."""
    # TODO: we need to do postprocessors in a clean way for each dataset.
    targets = [string_to_float(target) for target in targets]
    predictions= [string_to_float(prediction) for prediction in predictions]
    spearman_corrcoef = 100 * scipy.stats.spearmanr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(spearman_corrcoef):
        spearman_corrcoef = 0
    return {"spearmanr": spearman_corrcoef}


def f1_score_with_invalid(predictions, targets) -> dict:
    """Computes F1 score,  with any prediction != 0 or 1 is counted as incorrect.
    Args:
      targets: list of targets, either 0 or 1
      predictions: list of predictions, any integer value
    Returns:
      F1 score, where any prediction != 0 or 1 is counted as wrong.
    """
    def binary_reverse(labels):
       return ['0' if label == '1' else '1' for label in labels]
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    # Get indices of invalid predictions.
    invalid_idx_mask = np.logical_and(predictions != '0', predictions != '1')
    # For any prediction != 0 or 1, we set the prediction to the opposite of its corresponding target.
    predictions[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask])
    targets = targets.astype(np.int32)
    predictions = predictions.astype(np.int32)
    return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions)}

# TODO: maybe gaurd against invalid values https://stackoverflow.com/questions/56865344/how-do-i-calculate-the-matthews-correlation-coefficient-in-tensorflow
def matthews_corrcoef(predictions, targets) -> dict:
    """Computes the Matthews correlation coefficient."""
    targets = [string_to_float(target) for target in targets]
    predictions= [string_to_float(prediction) for prediction in predictions]
    matthews_correlation = 100 * sklearn.metrics.matthews_corrcoef(targets, predictions)

    # print(sum(predictions), sum(targets))
    if math.isnan(matthews_correlation):
        matthews_correlation = 0
    return {"matthews_correlation": matthews_correlation}

def squad(predictions, targets):
  """Computes SQuAD metrics, maximizing over answers per question.
  Args:
    targets: list of lists of strings
    predictions: list of strings
  Returns:
    dict with score_key: squad score across all targets and predictions
  """
  # FIXME: for multiple outputs; potentially dev""
  # targets = [[normalize_squad(t) for t in u] for u in targets]
  if isinstance(targets[0], str):
    targets = [[normalize_squad(u)] for u in targets]
  else:
    targets = [[normalize_squad(t) for t in u] for u in targets]
    
  predictions = [normalize_squad(p) for p in predictions]
  return qa_metrics(targets, predictions)

def exact_match(predictions, targets):
  """Computes whether the targets match predictions exactly."""
  return {"em": 100 * float(np.array_equal(targets, predictions))}


def sklearn_metrics_wrapper(metric_str,
                            metric_dict_str=None,
                            metric_post_process_fn=None,
                            **metric_fn_kwargs):
  """Wraps any sklearn.metric function and returns a t5 metric function.
  Args:
    metric_str: string, the function from `sklearn.metrics` to use.
    metric_dict_str: optional string, if not specified `metric_str` is used as
      the key in the returned dictionary.
    metric_post_process_fn: callable, if specified the final computed metric
      will be passed through this.
    **metric_fn_kwargs: kwargs, passed to the metric function we are calling.
  Returns:
    the function that calculates the metric in a dict.
  """
  if not hasattr(sklearn.metrics, metric_str):
    raise ValueError("sklearn.metrics does not have: %s" % metric_str)

  def fn(predictions, targets):
    metric_fn = getattr(sklearn.metrics, metric_str)
    metric_val = metric_fn(targets, predictions, **metric_fn_kwargs)
    if metric_post_process_fn is not None:
      metric_val = metric_post_process_fn(metric_val)
    return {metric_dict_str or metric_str: metric_val}
  return fn


def mean_multiclass_f1(num_classes, **metric_fn_kwargs):
  """Computes the unweighted average of the F1 per class."""
  return sklearn_metrics_wrapper(
      "fbeta_score",
      metric_dict_str="f1_multiclass",
      metric_post_process_fn=lambda x: 100 * x,
      beta=1,
      labels=range(num_classes),
      average="macro",
      **metric_fn_kwargs)


def multirc_f1_over_all_answers(targets, predictions):
  """Special metric for MultiRC which computes F1 score over all examples.
  This is necessary because the targets/predictions for MultiRC are dicts and
  the f1_score_with_invalid expects a list of True/False labels, not dicts. As
  a result we just need to key in the "value" for each of the example dicts
  before feeding into f1_score_with_invalid.
  Args:
    targets: list of dicts, where each dict has a "value" key.
    predictions: list of dicts, where each dict has a "value" key.
  Returns:
    F1 score over values, where any prediction != 0 or 1 is counted as wrong.
  """
  return f1_score_with_invalid(
      [t["value"] for t in targets], [p["value"] for p in predictions]
  )


def mean_group_metric(metric_fn, group_key="group", value_key="value"):
  """Returns a metric that averages `metric_fn` on sub-groups of results.
  The sub-groups are defined by aggregating results (targets and predictions)
  by accessing the feature specified by `group_key` in the target dicts.
  **WARNING**: Using this function can produce unreliable results if you do not
  pass in full groups. For example, if you evaluate over a random subsample of a
  validation set and do not retain all of the examples in each group, you may
  get results which aren't directly comparable to using the full validation set.
  Args:
    metric_fn: function, the metric to compute on the subgroups.
    group_key: string, the key for the grouping value in the target dictionary.
    value_key: string, the key for the value in the dictionaries.
  """
  def my_metric(targets, predictions):
    """Computes mean of `metric_fn` over subgroups of results."""
    grouped_values = collections.defaultdict(lambda: ([], []))
    for targ, pred in zip(targets, predictions):
      g = targ[group_key]
      grouped_values[g][0].append(targ[value_key])
      grouped_values[g][1].append(pred[value_key])
    group_scores = collections.defaultdict(list)
    for (targets, predictions) in grouped_values.values():
      for metric, score in metric_fn(targets, predictions).items():
        group_scores[metric].append(score)
    return {metric: np.mean(scores) for metric, scores in group_scores.items()}
  return my_metric


"""Utilities for Question Answering (QA) evaluation.
Matches results on the SQuAD (v1.1) and TriviaQA (v1.0) evaluation scripts.
"""

def _normalize_answer(text, punc_chars, punc_repl):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(s):
    return re.sub(r"\b(a|an|the)\b", " ", s)

  def replace_punctuation(s):
    to_replace = set(punc_chars)
    return "".join(punc_repl if ch in to_replace else ch for ch in s)

  def white_space_fix(s):
    return " ".join(s.split())

  text = text.lower()
  text = replace_punctuation(text)
  text = remove_articles(text)
  text = white_space_fix(text)
  return text


def normalize_trivia_qa(answer):
  """Normalization used in official TriviaQA evaluation script."""
  return _normalize_answer(
      answer, punc_chars=string.punctuation + "‘’´`_", punc_repl=" ").strip()


def normalize_squad(answer):
  """Normalization used in official SQuAD evaluation script."""
  return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl="")


def _metric_max_over_ground_truths(metric_fn, ground_truths, prediction):
  """Computes the maximum of the metric over all ground truths."""
  return max(
      metric_fn(ground_truth, prediction) for ground_truth in ground_truths
  )


def _exact_match_score(target, prediction):
  return target == prediction


def _f1_score(target, prediction):
  """Computes token f1 score for a single target and prediction."""
  prediction_tokens = prediction.split()
  target_tokens = target.split()
  common = (collections.Counter(prediction_tokens) &
            collections.Counter(target_tokens))
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(target_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def qa_metrics(targets, predictions):
  """Computes exact match and f1 QA scores, expecting pre-normalized text."""
  if len(targets) != len(predictions):
    raise ValueError("Number of targets and predictions must match.")
  em = np.mean([
      _metric_max_over_ground_truths(_exact_match_score, t, p)
      for p, t in zip(predictions, targets)
  ])
  f1 = np.mean([
      _metric_max_over_ground_truths(_f1_score, t, p)
      for p, t in zip(predictions, targets)
  ])
  em *= 100
  f1 *= 100
  return {"em": em, "f1": f1}