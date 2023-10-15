import os
import json
import torch
import pickle
from tqdm import tqdm
from collections import defaultdict

from torch.utils.data import Dataset

from utils import *


class DatasetCounter():
    # Keep tracking the counter of each dataset and return the next batch
    # TODO: shuffle the indices and return the indices for shuffled set
    def __init__(self, dataset_names, dataset_indices):
        self.dataset_names = dataset_names
        self.dataset_indices = dataset_indices
        
        # Assuming the indices are continuous for each dataset
        self.dataset_heads = list([x[0] for x in dataset_indices])
        self.dataset_tails = list([x[-1] for x in dataset_indices])
        
        self.dataset_counters = list(self.dataset_heads)  # put the counter to the first index of each dataset

    def get(self, task_name, sample_size):
        # Assumeing the task_name is a str
        cur_task_id = self.dataset_names.index(task_name)

        head, tail = self.dataset_heads[cur_task_id], self.dataset_tails[cur_task_id]
        cur_counter = self.dataset_counters[cur_task_id]

        if cur_counter + sample_size <= tail + 1:
            if self.dataset_counters[cur_task_id] + sample_size > tail:
                self.dataset_counters[cur_task_id] = head
            else:
                self.dataset_counters[cur_task_id] += sample_size
            cur_batch = list(range(cur_counter, cur_counter + sample_size))
            assert min(cur_batch) >= head and max(cur_batch) <= tail
            return cur_batch
        
        else:  # corner case
            turn_over = sample_size - (tail - cur_counter + 1)
            self.dataset_counters[cur_task_id] = head + turn_over
            cur_batch = list(range(cur_counter, tail + 1)) + list(range(head, head + turn_over))
            assert min(cur_batch) >= head and max(cur_batch) <= tail
            return cur_batch


class DatasetCounterShuffle():
    # Keep tracking the counter of each dataset and return the next batch
    def __init__(self, dataset_names, dataset_indices):
        self.dataset_names = dataset_names
        self.dataset_indices = dataset_indices
        
        for indices in self.dataset_indices:
            np.random.shuffle(indices)

        self.dataset_counters = [0] * len(self.dataset_names)

    def get(self, task_name, sample_size):
        # Assumeing the task_name is a str
        cur_task_id = self.dataset_names.index(task_name)
        head, tail = 0, len(self.dataset_indices[cur_task_id]) - 1
        cur_counter = self.dataset_counters[cur_task_id]

        if cur_counter + sample_size <= tail + 1:
            if self.dataset_counters[cur_task_id] + sample_size > tail:
                self.dataset_counters[cur_task_id] = head
            else:
                self.dataset_counters[cur_task_id] += sample_size
            cur_batch = list(range(cur_counter, cur_counter + sample_size))
            assert min(cur_batch) >= head and max(cur_batch) <= tail, f"{cur_batch} {cur_counter} {sample_size}"
        
        else:  # corner case
            turn_over = sample_size - (tail - cur_counter + 1)
            self.dataset_counters[cur_task_id] = head + turn_over
            cur_batch = list(range(cur_counter, tail + 1)) + list(range(head, head + turn_over))
            assert min(cur_batch) >= head and max(cur_batch) <= tail

        return [self.dataset_indices[cur_task_id][i] for i in cur_batch]


class CustomT5Dataset(Dataset):
    """
    The most plain dataloader; simply load the datasets and randomly sample it
    """

    def __init__(self, dataframe, tokenizer, source_len, target_len, datasets=[]):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.datasets = datasets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source_text = self.data[index]['source']
        target_text = self.data[index]['target']
        task_name = self.data[index]['task']
        task_id = self.datasets.index(task_name) if self.datasets else 0

        source = self.tokenizer(source_text,
                                max_length=self.source_len, 
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt'
                                )

        target = self.tokenizer(target_text,
                                max_length=self.target_len, 
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt'
                                )

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids']
        target_ids = torch.tensor([
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in target_ids
        ]).squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'task_ids': task_id,
            'extra_fields': str(self.data[index]['extra_fields']) if 'extra_fields' in self.data[index] else "{}"
        }

    

class CustomT5MultiDataset(Dataset):
    """
    Add sampling choices, t5 or uniform or unifiedqa
    """
    def __init__(self, dataframe, tokenizer, source_len, target_len, datasets=[], sampling_method='uniform', size_limit=2**18):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.datasets = datasets
        self.sampling_method = sampling_method

        if sampling_method == 'uniform':
            self.idx_mapping = list(range(len(self.data)))
        else:
            self.dataset_dict = defaultdict(list)
            for idx, data in enumerate(self.data):
                # self.dataset_dict[data['task']]
                self.dataset_dict[data['task']].append(idx)

            self.dataset_indices = [self.dataset_dict[x] for x in self.datasets]
            self.dataset_sizes = [len(self.dataset_dict[x]) for x in self.datasets]
            self.sample_distrib = dataset_sampling(self.dataset_sizes, self.sampling_method, size_limit=size_limit)

            self.reset()

    def reset(self, epoch=None):
        if self.sampling_method == 'uniform':
            self.idx_mapping = np.random.permutation(len(self.data))
        else:
            all_indices = []
            for i, d in enumerate(self.datasets[:-1]):
                new_size = int(self.sample_distrib[i] * len(self.data))
                if new_size > len(self.dataset_indices[i]):
                    indices = np.random.choice(self.dataset_indices[i], new_size).tolist()
                else:
                    # do not sample duplicates when the new_size is smaller than the dataset size
                    indices = np.random.choice(self.dataset_indices[i], new_size, replace=False).tolist()

                all_indices += indices

            new_size = len(self.data) - len(all_indices)
            if new_size > len(self.dataset_indices[-1]):
                indices = np.random.choice(self.dataset_indices[-1], new_size).tolist()
            else:
                indices = np.random.choice(self.dataset_indices[-1], new_size, replace=False).tolist()
            all_indices += indices

            self.idx_mapping = np.random.permutation(all_indices)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        data_idx = self.idx_mapping[index]
        data = self.data[int(data_idx)]

        source_text = data['source']
        target_text = data['target']
        task_name = data['task']
        task_id = self.datasets.index(task_name) if self.datasets else 0

        source = self.tokenizer(source_text,
                                max_length=self.source_len, 
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt'
                                )

        target = self.tokenizer(target_text,
                                max_length=self.target_len, 
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt'
                                )

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids']
        target_ids = torch.tensor([
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in target_ids
        ]).squeeze()

        return {
            'data_idx': data_idx, 
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'task_ids': task_id,
            'extra_fields': str(data['extra_fields']) if 'extra_fields' in self.data[index] else "{}"
        }



class CustomStochTaskDataset(Dataset):
    """
    Add stochastic task sampling - dynamically change the #tasks per batch
    """

    def __init__(self, dataframe, tokenizer, source_len, target_len, datasets=[], 
                 sampling_method='uniform', batch_size=36, size_limit=2**18):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.datasets = datasets
        self.sampling_method = sampling_method
        self.batch_size = batch_size

        self.length = len(self.data) // batch_size

        self.dataset_dict = defaultdict(list)
        for idx, data in tqdm(enumerate(self.data), total=len(self.data), disable=True):
            self.dataset_dict[data['task']].append(idx)

        self.dataset_indices = [self.dataset_dict[x] for x in self.datasets]
        self.dataset_sizes = [len(self.dataset_dict[x]) for x in self.datasets]
        self.sample_distrib = dataset_sampling(self.dataset_sizes, self.sampling_method, size_limit=size_limit)

        self.reset(0)
 
    def reset(self, epoch=None):

        DataCounter = DatasetCounterShuffle(self.datasets, self.dataset_indices)

        self.batch_mapping = [] # list of batches
        task_numbers = range(2, len(self.datasets) + 1)

        for i in range(self.length):

            task_number = np.random.choice(task_numbers)

            cur_batch = []
            if self.batch_size <= task_number:

                tasks = np.random.choice(self.datasets, self.batch_size, p=self.sample_distrib)  # task sampling; replace=True

                for task in tasks:
                    temp_idx = DataCounter.get(task, 1)
                    cur_batch += temp_idx

            else:
                tasks = np.random.choice(self.datasets, task_number, p=self.sample_distrib)  # task sampling; replace=True

                pertask = self.batch_size // task_number
                lefts = self.batch_size % task_number
                
                for task in tasks:
                    temp_idx = DataCounter.get(task, pertask)
                    cur_batch += temp_idx

                if lefts:
                    temp_idx = DataCounter.get(np.random.choice(tasks), lefts)
                    cur_batch += temp_idx

            assert len(cur_batch) == self.batch_size
            np.random.shuffle(cur_batch)
            self.batch_mapping.append(cur_batch)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        batch_data = self.batch_mapping[index]
        batch_data = [self.data[x] for x in batch_data]

        batch_source_ids = []
        batch_source_mask = []
        batch_target_ids = []
        batch_task_ids = []
        batch_extra_fields = []
        for data in batch_data:
            source_text = data['source']
            target_text = data['target']
            task_name = data['task']
            task_id = self.datasets.index(task_name) if self.datasets else 0

            source = self.tokenizer(source_text,
                                    max_length=self.source_len, 
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt'
                                    )

            target = self.tokenizer(target_text,
                                    max_length=self.target_len, 
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt'
                                    )

            source_ids = source['input_ids']
            source_mask = source['attention_mask']
            target_ids = target['input_ids']
            target_ids = torch.tensor([
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in target_ids
            ])
            extra_fields = str(data['extra_fields']) if 'extra_fields' in self.data[index] else "{}"

            batch_source_ids.append(source_ids)
            batch_source_mask.append(source_mask)
            batch_target_ids.append(target_ids)
            batch_task_ids.append(task_id)
            batch_extra_fields.append(extra_fields)
        
        batch_source_ids = torch.cat(batch_source_ids, dim=0).to(dtype=torch.long)
        batch_source_mask = torch.cat(batch_source_mask, dim=0).to(dtype=torch.long)
        batch_target_ids = torch.cat(batch_target_ids, dim=0).to(dtype=torch.long)

        return {
            # 'data_idx': data_idx, 
            'source_ids': batch_source_ids, 
            'source_mask': batch_source_mask, 
            'target_ids': batch_target_ids,
            'task_ids': batch_task_ids,
            'extra_fields': batch_extra_fields
        }