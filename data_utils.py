# Handling dataset loading and preprocessing

import os
import re
import sys
import abc
import random
import logging
import functools
import numpy as np
import collections
from collections import OrderedDict
from typing import Callable, List, Mapping

import torch
import datasets

import metrics
from utils import round_stsb_target, pad_punctuation

from datasets.utils import disable_progress_bar
disable_progress_bar()

logger = logging.getLogger(__name__)

class AbstractTask(abc.ABC):
    name = NotImplemented
    config = NotImplemented
    prefix = NotImplemented
    preprocessor: Callable = NotImplemented
    metric = NotImplemented
    metric_names = NotImplemented
    split_map = None
    labels_list = None
    split_to_data_split: Mapping[str, str] = {"train": "train", "validation": "validation", "test": "test"}
    small_datasets_without_all_splits = ["cola", "wnli", "rte", "superglue-cb", "superglue-copa", "superglue-multirc",
                                         "superglue-wic", "superglue-wsc.fixed", "superglue-rte", "mrpc", "stsb",
                                         "superglue-boolq", "xsum", "scitail"]
    large_data_without_all_splits = ["qqp", "qnli", "superglue-record", "sst2", "squad", "snli", "anli",
                                     "amazon_polarity", "yelp_polarity", "winogrande", "newsqa", "searchqa", "triviaqa", "naturalquestions", "hotpotqa"]

    def __init__(self, config, seed=42):
        self.config = config
        self.seed = seed

    def get_max_target_length(self, tokenizer, default_max_length):
        if self.labels_list is not None:
            return max([len(tokenizer.encode(label)) for label in self.labels_list])
        return default_max_length

    def seq2seq_format(self, sources: List[str],
                       targets: List[str],
                       add_prefix: bool = False,
                       prefix: str = None,
                       extra_fields={},
                       verbalizer=""):
        
        src_prefix = self.name if prefix is None else prefix
        if verbalizer:
            sources = [verbalizer] + sources
        # sources = [src_prefix] + sources if add_prefix else sources
        if len(extra_fields) == 0:
            return {'source': ' '.join(sources),
                    'target': ' '.join(targets),
                    'task': self.name}
        else:
            return {'source': ' '.join(sources),
                    'target': ' '.join(targets),
                    'task': self.name,
                    'extra_fields': extra_fields}

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        
        if indices is None:
            indices = self.shuffled_indices(dataset)
            
        indices = indices[:n_obs]
        
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, self.config, split=split, script_version="master")

    def get_split_indices(self, dataset):
        indices = self.shuffled_indices(dataset)
        # change by Yao
        return indices
       
    def map_dataset(self, dataset, add_prefix, add_vb):
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix, add_vb=add_vb),
                           remove_columns=dataset.column_names, load_from_cache_file=False)

    def get(self, split, add_prefix=True, n_obs=None, split_validation_test=False, lang=None, file_name=None, add_vb=False, file_prefix=None):
        self.file_prefix = file_prefix  # path prefix for external dataset loading

        if split_validation_test and split == "train":
            mapped_split = self.split_to_data_split["train"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_split_indices(dataset)
            dataset = self.subsample(dataset, n_obs, indices)
        else:
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_split_indices(dataset)
            dataset = self.subsample(dataset, n_obs, indices)

        return self.map_dataset(dataset, add_prefix, add_vb)
    

class Squad(AbstractTask):
    name = "squad"
    metric = [metrics.squad]

    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset("squad", split=split)

    def preprocessor(self, example, add_prefix, add_vb=False):
        answer = pad_punctuation(example['answers']['text'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)


class NewsQA(AbstractTask):
    name = "newsqa"
    metric = [metrics.squad]

    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    split_to_file_name = {
        'train': 'NewsQA_train',
        'validation': 'NewsQA_dev'
    }

    def load_dataset(self, split):
        return datasets.load_dataset('json', data_files={
            split: os.path.join(self.file_prefix, f'{self.split_to_file_name[split]}.jsonl')
        })[split]

    def preprocessor(self, example, add_prefix, add_vb=False):
        answer = pad_punctuation(example['answers']['text'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)


class SearchQA(AbstractTask):
    name = "searchqa"
    metric = [metrics.squad]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    split_to_file_name = {
        'train': 'SearchQA_train',
        'validation': 'SearchQA_dev'
    }

    def load_dataset(self, split):
        return datasets.load_dataset('json', data_files={
            split: os.path.join(self.file_prefix, f'{self.split_to_file_name[split]}.jsonl')
        })[split]

    def preprocessor(self, example, add_prefix, add_vb=False):
        answer = pad_punctuation(example['answers']['text'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)


class NaturalQA(AbstractTask):
    name = "naturalquestions"
    metric = [metrics.squad]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    split_to_file_name = {
        'train': 'NaturalQuestionsShort_train',
        'validation': 'NaturalQuestionsShort_dev'
    }

    def load_dataset(self, split):
        return datasets.load_dataset('json', data_files={
            split: os.path.join(self.file_prefix, f'{self.split_to_file_name[split]}.jsonl')
        })[split]

    def preprocessor(self, example, add_prefix, add_vb=False):
        answer = pad_punctuation(example['answers']['text'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)


class HotpotQA(AbstractTask):
    name = "hotpotqa"
    metric = [metrics.squad]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    split_to_file_name = {
        'train': 'HotpotQA_train',
        'validation': 'HotpotQA_dev'
    }

    def load_dataset(self, split):
        return datasets.load_dataset('json', data_files={
            split: os.path.join(self.file_prefix, f'{self.split_to_file_name[split]}.jsonl')
        })[split]

    def preprocessor(self, example, add_prefix, add_vb=False):
        answer = pad_punctuation(example['answers']['text'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)


class SciTail(AbstractTask):
    name = "scitail"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('scitail', "snli_format", split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        label2id = {"entailment": "0", "neutral": "1"}
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [label2id[example["gold_label"]]]
        if add_vb:
            verbalizer = "{ 0 : entailment, 1 : neutral }"	
        else:
            verbalizer = ""	
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)


class MRPC(AbstractTask):
    name = "mrpc"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy, metrics.f1_score_with_invalid]
    metric_names = ["accuracy", "f1"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        # return datasets.load_dataset('glue', 'mrpc', split=split, script_version="master")
        return datasets.load_dataset('glue', 'mrpc', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        if add_vb:
            verbalizer = "{ 0 : not equivalent, 1 : equivalent }"
        else:
            verbalizer = ""
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)


class COLA(AbstractTask):
    name = "cola"
    labels_list = ["0", "1"]
    metric = [metrics.matthews_corrcoef]
    metric_names = ["matthews_correlation"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):	
        src_texts = ["sentence:", example['sentence']]	
        tgt_texts = [str(example['label'])]	
        if add_vb:	
            verbalizer = "{ 0 : grammatically unacceptable, 1 : grammatically acceptable }"	
        else:	
            verbalizer = ""	
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)

class SST2(AbstractTask):
    name = "sst2"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        if add_vb:
            verbalizer = "{ 0 : negative, 1 : positive }"
        else:
            verbalizer = ""
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)


class YelpPolarity(AbstractTask):
    name = "yelp_polarity"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "test": "test"}

    split_to_file_name = {
        'train': 'yelp_train',
        'test': 'yelp_test'
    }

    def load_dataset(self, split):
        return datasets.load_dataset('csv', data_files={
            split: os.path.join(self.file_prefix, f'{self.split_to_file_name[split]}.csv')},
            column_names=['label', 'text']
            )[split]

    # def load_dataset(self, split):
    #     return datasets.load_dataset('yelp_polarity')[split]

    def preprocessor(self, example, add_prefix=True,  add_vb=False):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'] - 1)]  # labels from local files
        if add_vb:
            verbalizer = "{ 0 : negative, 1 : positive }"
        else:
            verbalizer = ""
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)


class Amazon_Polarity(AbstractTask):
    name = "amazon_polarity"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('yelp_polarity', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", "<title> {0} <context> {1}".format(
            example['title'], example['context'])]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class STSB(AbstractTask):
    name = "stsb"
    labels_list = [str(np.round(label, decimals=1))
                   for label in np.arange(0, 5.2, 0.2)]
    metric = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    metric_names = ["pearson", "spearmanr"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QQP(AbstractTask):
    name = "qqp"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy, metrics.f1_score_with_invalid]
    metric_names = ["accuracy", "f1"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["question1:", example['question1'],
                     "question2:", example["question2"]]
        tgt_texts = [str(example['label'])]
        if add_vb:
            verbalizer = "{ 0 : not duplicate, 1 : duplicate }"
        else:
            verbalizer = ""
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["premise:", example['premise'],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        if add_vb:
            verbalizer = "{ 0 : entailment, 1 : neutral, 2 : contradiction }"
        else:
            verbalizer = ""
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)


class SNLI(AbstractTask):
    name = "snli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('snli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis: ", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MultiNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('multi_nli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class ANLI(AbstractTask):
    name = "anli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train_r3",
                           "validation": "dev_r3",
                           "test": "test_r3"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        split_to_data_split = {"train": "train_r3",
                               "validation": "dev_r3",
                               "test": "test_r3"}
        return datasets.load_dataset('anli', split=split_to_data_split[split] if split in split_to_data_split else split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["question:", example['question'],
                     "sentence:", example["sentence"]]
        tgt_texts = [str(example['label'])]
        if add_vb:
            verbalizer = "{ 0 : entailment, 1 : not entailment }"
        else:
            verbalizer = ""
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)

class RTE(AbstractTask):
    name = "rte"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        if add_vb:
            verbalizer = "{ 0 : entailment, 1 : not entailment }"
        else:
            verbalizer = ""
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)


class WNLI(AbstractTask):
    name = "wnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'wnli', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        if add_vb:
            verbalizer = "{ 0: not entailment, 1 : entailment }"
        else:
            verbalizer = ""
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)


class SuperGLUEBoolQ(AbstractTask):
    name = "superglue-boolq"
    labels_list = ['0', '1']
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'boolq', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["question:", example["question"],
                     "passage:", example["passage"]]
        tgt_texts = [str(example["label"])]
        if add_vb:
            verbalizer = "{ 0 : false, 1 : true }"
        else:
            verbalizer = ""
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)


class SuperGLUERTE(AbstractTask):
    name = "superglue-rte"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'rte', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUECB(AbstractTask):
    name = "superglue-cb"
    labels_list = ['0', '1', '2']
    # labels_list = ['entailment', 'contradiction', 'neutral']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.mean_multiclass_f1(num_classes=3), metrics.accuracy]
    metric_names = ["f1_multiclass", "accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'cb', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]

        tgt_texts = [str(example["label"])]
        if add_vb:
            verbalizer = "{ 0 : entailment, 1 : contradiction, 2 : neutral }"
        else:
            verbalizer = ""
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)


class SuperGLUECOPA(AbstractTask):
    name = "superglue-copa"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'copa', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["premise:", example["premise"],
                     "choice1:", example["choice1"],
                     "choice2:", example["choice2"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUEMultiRC(AbstractTask):
    name = "superglue-multirc"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.multirc_f1_over_all_answers,
              metrics.mean_group_metric(metrics.exact_match)]
    metric_names = ["f1", "em"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'multirc', split=split)

    def remove_markup(self, text):
        """Removes the HTML markup."""
        text = re.sub('<br>', ' ', text)
        text = re.sub('<(/)?b>', '', text)
        return text

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        group = example['idx']['question']
        # T5 applies remove_markup to the joined string, but this should not make
        # any difference as well.
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/data/preprocessors.py#L797
        src_texts = ["question:", self.remove_markup(example["question"]),
                     "answer:", self.remove_markup(example["answer"]),
                     "paragraph:", self.remove_markup(example["paragraph"])]
        tgt_texts = [str(example["label"])]
        if add_vb:
            verbalizer = "{ 0 : false, 1 : true }"
        else:
            verbalizer = ""
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields={"group": group}, verbalizer=verbalizer)


class SuperGLUEWIC(AbstractTask):
    name = "superglue-wic"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'wic', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["sentence1:", example["sentence1"],
                     "sentence2:", example["sentence2"],
                     "word:", example["word"]]
        tgt_texts = [str(example["label"])]
        if add_vb:
            verbalizer = "{ 0 : false, 1 : true }"
        else:
            verbalizer = ""
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)


class SuperGLUEWSCFixed(AbstractTask):
    # source: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
    """Convert WSC examples to text2text format.
     WSC includes a sentence along with 2 'spans': the first denoting a noun and
     the other a pronoun. The 'label' specifies whether or not the pronoun is
     referencing the noun. This preprocessor puts ' * ' around the noun and ' # '
     around the pronoun.
     For example, a typical example from WSC might look like
     {
         'text': 'This is a test sentence .',
         'span1_text': 'test',
         'span1_index': 3,
         'span2_text': 'This',
         'span2_index': 0,
         'label': 0
     }
     This example would be transformed to
     {
         'inputs': 'wsc text: # This # is a * test * sentence .',
         'targets': 'False'
     }
    """
    name = "superglue-wsc.fixed"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]


    def get(self, split, add_prefix=False, n_obs=None, split_validation_test=False, lang=None, file_name=None, add_vb=False, file_prefix=None):
        self.file_prefix = file_prefix  # path prefix for external dataset loading

        if split_validation_test and split == "train":
            mapped_split = self.split_to_data_split["train"]
            dataset = self.load_dataset(split=mapped_split)
            dataset = dataset.filter(lambda example: example["label"]==1)
            indices = self.get_split_indices(dataset)
            dataset = self.subsample(dataset, n_obs, indices)
        else:
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            dataset = dataset.filter(lambda example: example["label"]==1)
            indices = self.get_split_indices(dataset)
            dataset = self.subsample(dataset, n_obs, indices)

        return self.map_dataset(dataset, add_prefix, add_vb)

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'wsc.fixed', split=split)

    def _mark_span(self, text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:\S+\s){N})(W)'
        pattern = re.sub('N', str(span_idx), pattern_tmpl)
        pattern = re.sub('W', span_str, pattern)
        return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

    def preprocessor(self, example, add_prefix=True, add_vb=False, is_training=False):
        # converts text as done in T5.
        text = example['text']
        span2_index = example['span2_index'] + 0 * int(example['span1_index'] < example['span2_index'])
        text = self._mark_span(text, example['span2_text'], span2_index, '*')
        
        src_texts = [text]
        tgt_texts = [example['span1_text']]
        
        if add_vb:
            verbalizer = "{ 0 : false, 1 : true }"
        else:
            verbalizer = ""
        
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)
    


class SuperGLUERecord(AbstractTask):
    """Convert ReCoRD examples to text2text examples.
    ReCoRD contains a passage, query containing a '@placeholder' string, and a set
    of entities that are the possible values of the placeholder. Each train and
    validation example will have a list of answers, any of which would be
    considered correct.
    For example, a typical example from ReCoRD might look like
    {
      'passsage': 'This is the passage.',
      'query': 'A @placeholder is a bird.',
      'entities': ['penguin', 'potato', 'pigeon'],
      'answers': ['penguin', 'pigeon'],
    }
    which this preprocessor would turn into the following two examples:
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'penguin',
    }
    and
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'pigeon',
    }
    """
    name = "superglue-record"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.squad]
    metric_names = ["squad"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'record', split=split)

    def preprocessor(self, batch, add_prefix=True, add_vb=False):
        new_batch = collections.defaultdict(list)
        keys = batch.keys()
        for values in zip(*batch.values()):
            ex = {k: v for k, v in zip(keys, values)}
            # updates the passage.
            passage = ex['passage']
            passage = re.sub(
                r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
            passage = re.sub(r'\n@highlight\n', '. ', passage)
            inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
            if add_prefix:
                inputs = self.name + " " + inputs
            # duplicates the samples based on  number of answers.
            num_answers = len(ex["answers"])
            num_duplicates = np.maximum(1, num_answers)
            new_batch["source"].extend([inputs] * num_duplicates)
            new_batch["target"].extend(
                ex["answers"] if num_answers > 0 else ["<unk>"])
            new_batch["task"].extend([self.name] * num_duplicates)
            new_batch["extra_fields"].extend(
                [{"answers": ex["answers"]}]*num_duplicates)
        return new_batch

    def map_dataset(self, dataset, add_prefix=True, add_vb=False):
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           batched=True, remove_columns=dataset.column_names, load_from_cache_file=False)


class IMDB(AbstractTask):
    name = "imdb"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "test"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        if split == "validation":
            split = "test"
        return datasets.load_dataset('imdb', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WinoGrande(AbstractTask):
    name = "winogrande"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('winogrande', "winogrande_xl", split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["sentence:", example["sentence"],
                     "option0:", example["option1"],
                     "option1:", example["option2"]]
        tgt_texts = [str(int(example["answer"]) - 1)]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class PAWS(AbstractTask):
    name = "paws"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('paws', 'labeled_final', split=split)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        if add_vb:
            verbalizer = "{ 0 : not paraphrase, 1 : paraphrase }"
        else:
            verbalizer = ""
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, verbalizer=verbalizer)


TASK_MAPPING = OrderedDict(
    [
        ('squad', Squad),
        ('mrpc', MRPC),
        ('cola', COLA),
        ('sst2', SST2),
        ('qnli', QNLI),
        ('rte', RTE),
        ('wnli', WNLI),
        ('mnli', MNLI),
        ('qqp', QQP),
        ('stsb', STSB),
        ('superglue-boolq', SuperGLUEBoolQ),
        ('superglue-rte', SuperGLUERTE),
        ('superglue-cb', SuperGLUECB),
        ('superglue-copa', SuperGLUECOPA),
        ('superglue-multirc', SuperGLUEMultiRC),
        ('superglue-wic', SuperGLUEWIC),
        ('superglue-wsc.fixed', SuperGLUEWSCFixed),
        ('superglue-record', SuperGLUERecord),
        ('multi_nli', MultiNLI),
        ('snli', SNLI),
        ('newsqa', NewsQA),
        ('searchqa', SearchQA),
        ('naturalquestions', NaturalQA),
        ('hotpotqa', HotpotQA),
        ('triviaqa', Squad),
        ('anli', ANLI),
        ("imdb", IMDB),
        ("winogrande", WinoGrande),
        ("scitail", SciTail),
        ('yelp_polarity', YelpPolarity),
        ('amazon_polarity', Amazon_Polarity),
        ('paws', PAWS),
    ]
)


class AutoTask:
    @classmethod
    def get(self, task, config='en', seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
