# import re
import regex as re
import os
import torch
import random
import numpy as np
from pathlib import Path


TASK_TO_METRICS = {"mrpc": ["accuracy", "f1"],
                   "cola": ['matthews_correlation'],
                   "stsb": ['pearson', 'spearmanr'],
                   'sst2': ['accuracy'],
                   "mnli": ["accuracy"],
                   "mnli_mismatched": ["accuracy"],
                   "mnli_matched": ["accuracy"],
                   "qnli": ["accuracy"],
                   "rte": ["accuracy"],
                   "wnli": ["accuracy"],
                   "qqp": ["accuracy", "f1"],
                   "superglue-boolq": ["accuracy"],
                   "superglue-rte": ["accuracy"],
                   "superglue-cb": ["accuracy", "f1_multiclass"],
                   "superglue-copa": ["accuracy"],
                   "superglue-multirc": ["f1", "em"],
                   "superglue-wic": ["accuracy"],
                   "superglue-wsc.fixed": ["accuracy"],
                   "superglue-record": ["f1", "em"],
                   "multi_nli": ["accuracy"],
                   "squad": ["f1", "em"],
                   "snli": ["accuracy"],
                   "naturalquestions": [ "f1", "em"],
                   "hotpotqa": ["f1", "em"],
                   "searchqa": ["f1", "em"],
                   "newsqa": ["f1", "em"],
                   "triviaqa": ["f1", "em"],
                   "imdb": ["accuracy"],
                   "winogrande": ["accuracy"],
                   "scitail": ["accuracy"],
                   "amazon_polarity": ["accuracy"],
                   "yelp_polarity": ["accuracy"],
                   "paws": ["accuracy"],}


TASK_NAME_MAPPING = {"mrpc": "mrpc",
                   "cola": "cola",
                   "stsb": "stsb",
                   'sst2': "sst2",
                   "mnli": "mnli",
                   "qnli": "qnli",
                   "rte": "rte",
                   "wnli": "wnli",
                   "qqp": "qqp",
                   "boolq": "superglue-boolq",
                   "cb": "superglue-cb",
                   "copa": "superglue-copa",
                   "multirc": "superglue-multirc",
                   "wic": "superglue-wic",
                   "wsc": "superglue-wsc.fixed",
                   "record": "superglue-record",
                   "squad": "squad",
                   "nq": "naturalquestions",
                   "hotpotqa": "hotpotqa",
                   "searchqa": "searchqa",
                   "newsqa": "newsqa",
                   "winogrande": "winogrande",
                   "scitail": "scitail",
                   "yelp": "yelp_polarity",
                   "paws": "paws"
                   }

def dataset_sampling(sizes, name='uniform', smooth_factor=0.001, size_limit=2**18):

  if name == 'uniform':
    return [x / sum(sizes) for x in sizes]

  elif name == 'stratified':
    base = sum([np.log(x) for x in sizes])
    return [(np.log(x) + smooth_factor) / (base + smooth_factor) for x in sizes]
  
  elif name == 't5':
    base = sum([min(x, size_limit) for x in sizes])
    return [min(x, size_limit) / base for x in sizes]

  elif name == 'unifiedqa':
    return [1 / len(sizes) for _ in sizes]

  else:
    raise Exception('Wrong Sampling Strategy!')



def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def save_checkpoint(path, checkpoint, filename="soft_prompt"):
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, os.path.join(path, filename))
    # print(f"Saved parameters at: {os.path.join(path, filename)}")


def generate_sharing_matrix(num_tasks, num_prompts, sharing_ratio, is_dynamic=False, seed=42):
  """
  Depending on the sharing ratio and whether we want to enable dynamic sharing
  1. No dynamic, ratio=0: ind. param. for each task
  2. No dynamic, ratio=1: sharing the same set of param. for all tasks
  3. No dynamic, 0<ratio<1: sharing partial params for all tasks, each task has other ind. params
  4. Dynamic (ratio is irrelevant): random sample indices for each task with a fixed number 
  Return an index matrix to retrieve the corresponding prompts for specific tasks
  -> assume this is the embedding ids for an embedding layer with the shape of [T, T*P]
  """
  # np.random.seed(seed)
  Tn, Pn = num_tasks, num_prompts
  index_mat = np.zeros((Tn, Tn * Pn))
  if is_dynamic:
    for ti in range(Tn):
      indices = np.random.permutation(range(Tn * Pn))[:Pn]
      index_mat[ti, indices] = 1
  else:
    if sharing_ratio == 0:
      # sharing the corresponding row of each patch
      for ti in range(Tn):
        start, end = Pn * ti, Pn * (ti + 1)
        index_mat[ti, start: end] = 1
    elif sharing_ratio == 1:
      # sharing the first patch
      index_mat[:, 0:Pn] = 1
    else:
      assert 0 < sharing_ratio < 1
      # mix the above cases
      Sn = int(Pn * sharing_ratio)
      index_mat[:, 0: Sn] = 1
      for ti in range(Tn):
        start, end = Sn + Pn * ti, Pn * (ti + 1)
        index_mat[ti, start: end] = 1

  row_indices = [np.nonzero(index_mat[i])[0].tolist() for i in range(Tn)]
  allnonzeros = list(set(sum(row_indices, [])))

  shares = []
  for i in range(Tn):
    for j in range(i+1, Tn):
      shares.append(len(set(row_indices[i]) & set(row_indices[j])) / Pn)

  return row_indices, allnonzeros, np.mean(shares)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def round_stsb_target(label):
    """STSB maps two sentences to a floating point number between 1 and 5
    representing their semantic similarity. Since we are treating all tasks as
    text-to-text tasks we need to convert this floating point number to a string.
    The vast majority of the similarity score labels in STSB are in the set
    [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
    entry in this set, and then we convert the result to a string (literally e.g.
    "3.4"). This converts STSB roughly into a 26-class classification dataset.
    Args:
      label: original label.
    Returns:
      A preprocessed label.
    """
    return np.round((label * 5) / 5, decimals=1)


def string_to_float(string, default=-1., **unused_kwargs):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


def pad_punctuation(text):
   """Re-implementation of _pad_punctuation in t5. This function adds spaces
   around punctuation. While this pads punctuation as expected, it has the 
   unexpected effected of padding certain unicode characters with accents, with
   spaces as well. For instance: "François" becomes "Fran ç ois"""
   # Pad everything except for: underscores (_), whitespace (\s),
   # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
   text = re.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', str(text))
   # Collapse consecutive whitespace into one space.
   text = re.sub(r'\s+', ' ', text)
   return text