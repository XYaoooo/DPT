import os
import sys
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import transformers
from transformers import (
    AutoTokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    AutoConfig,
    AdamW,
    get_scheduler,
    set_seed,
)
transformers.logging.set_verbosity_error()

from modeling_t5 import T5PromptForConditionalGeneration_param
from data_utils import AutoTask
from eval_utils import AutoPostProcessor
from metrics import *
from options import *
from utils import *
from loader import *

import datasets
from datasets import concatenate_datasets
from datasets.utils.logging import set_verbosity_error
from eval_utils import *
set_verbosity_error()

import logging
logging.disable(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")


def run(local_rank, args):

    is_master = local_rank == 0
    world_size = args.world_size
    is_mp = world_size > 1
    # set the device
    device = local_rank

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    # Can processing multiple datasets
    train_datasets, val_datasets = [], {}
    for dataset_name in args.datasets_names:
        train_dataset = AutoTask.get(dataset_name).get(split="train",
                                    split_validation_test=True,
                                    add_prefix=args.add_task_prefix,
                                    add_vb=args.add_verbalizer,
                                    file_prefix=args.local_file_prefix,
                                    n_obs=100000 if dataset_name == 'yelp_polarity' else None)
        train_datasets.append(train_dataset)

        val_dataset = AutoTask.get(dataset_name).get(split="validation",
                                    split_validation_test=True,
                                    add_prefix=args.add_task_prefix,
                                    add_vb=args.add_verbalizer,
                                    file_prefix=args.local_file_prefix,
                                    n_obs=None)
        val_datasets.update({dataset_name: val_dataset})

        
        if is_master:
            print(local_rank, dataset_name, 'Train\t', train_dataset[0])
            print(local_rank, dataset_name, 'Val\t', val_dataset[0])
            
    # merge all datasets if there are multiple ones
    train_datasets = concatenate_datasets(train_datasets)

    if is_master:
        print(local_rank, len(train_dataset), len(val_dataset))
        print('# all training samples:', len(train_datasets))
        print(train_datasets[0])

    # Data loader 
    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomT5Dataset(train_datasets, tokenizer, 
                                    args.max_source_length, 
                                    args.max_target_length, 
                                    args.datasets_names
                                    )
    

    val_sets = {data_name: CustomT5Dataset(data_set, tokenizer, 
                                           args.max_source_length, 
                                           args.max_target_length, 
                                           args.datasets_names
                                           )
               for data_name, data_set in val_datasets.items()}

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': args.train_batch_size,
        'shuffle': not is_mp,  # not shuffle in DDP
        'num_workers': 4,
        'worker_init_fn': seed_worker,
        'generator': g,
        }

    eval_params = {
        'batch_size': args.eval_batch_size,
        'shuffle': False,
        'num_workers': 0,
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    if is_mp:
        sampler = DistributedSampler(training_set, num_replicas=world_size, rank=local_rank, shuffle=True)
        TrainDataloader = DataLoader(training_set, sampler=sampler, **train_params)
    else:
        TrainDataloader = DataLoader(training_set, **train_params)

    ValDataloaders = {data_name: DataLoader(data_set, **eval_params) for data_name, data_set in val_sets.items()}

    ###################################################################################################
    #   Build the model                                                                                   #
    ###################################################################################################
    config = AutoConfig.from_pretrained(args.model_name)
    config.len_enc_prompt = args.enc_prompt_tokens
    config.len_dec_prompt = args.dec_prompt_tokens
    config.add_enc_prompt = args.enc_prompt_tokens > 0
    config.add_dec_prompt = args.dec_prompt_tokens > 0
    config.num_tasks = len(args.datasets_names)
    
    config.bottle_neck = args.bottle_neck

    model = T5PromptForConditionalGeneration_param.from_pretrained(args.model_name, config=config)

    # Freeze the backbone model
    for name, param in model.named_parameters():
        param.requires_grad = False if 'prefix' not in name else True

    if is_mp:
        # initialize distributed data parallel (DDP)
        dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

        model = DDP(
            model.to(local_rank),
            device_ids=[local_rank],
            find_unused_parameters=False
        )
    else:
        model = model.to(device)

    
    if is_master:
        print('Parameters to optimize: ', [n for n, p in model.named_parameters() if 'prefix' in n])
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if 'prefix' in n],
            "weight_decay": args.weight_decay,
        }
    ]
 
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr * np.sqrt(world_size))  # sacle the learning rate based on world_size

    max_train_steps = args.max_train_steps if args.max_train_steps > 0 else args.n_epochs * len(TrainDataloader)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.num_warmup_steps) if args.num_warmup_steps > 1 else int(args.num_warmup_steps * max_train_steps),
        num_training_steps=max_train_steps
    )

    # Load the model or resume the training
    resume_steps = 0
    if args.from_checkpoint:
        if is_mp:
            checkpoint = torch.load(args.from_checkpoint, map_location=torch.device(f'cuda:{local_rank}'))
        else:
            checkpoint = torch.load(args.from_checkpoint)

        resume_steps = checkpoint['global_step']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        param_dict = checkpoint['params']
        for n, p in model.named_parameters():
            if n in param_dict:
                p.data = param_dict[n].clone().detach().to(device)

        if is_master:
            print('Resume the training from the checkpoint: ', args.from_checkpoint)

    if args.auto_resume and args.save_ckpt_dir:
        checkpoint_path = os.path.join(args.save_ckpt_dir, args.latest_ckpt_name)

        if os.path.exists(checkpoint_path) or len(os.listdir(args.save_ckpt_dir)) > 0:
            if not os.path.exists(checkpoint_path):
                list_files = os.listdir(args.save_ckpt_dir)
                # little parsing to get the step number: format -> sst2.qqp.mnli.qnli.squad.record.soft_prompts.source.step.900.pt
                list_steps = [x.strip().split('.')[-2] for x in list_files]
                max_idx = list_steps.index(max(list_steps))
                checkpoint_path = os.path.join(args.save_ckpt_dir, list_files[max_idx])    

            assert os.path.exists(checkpoint_path)

            if is_mp:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device(f'cuda:{local_rank}'))
            else:
                checkpoint = torch.load(checkpoint_path)

            resume_steps = checkpoint['global_step']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            param_dict = checkpoint['params']
            for n, p in model.named_parameters():
                if n in param_dict:
                    p.data = param_dict[n].clone().detach().to(device)

            if is_master:
                print(f'Auto-resume the training from the checkpoint: {checkpoint_path} from step {resume_steps}')
        else:
            if is_master:
                print('No existing checkpoint; Start the training from scratch!')

    
    if is_master:
        print('Prefix Parameters: ', [n for n, p in model.named_parameters() if 'prefix' in n])
        print('Trainable Parameters: ', [n for n, p in model.named_parameters() if p.requires_grad])
        print('#Trainable Parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

        if args.prompt_type != 'dynamic':
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'All trainable parameters: {trainable_params}; per task: {trainable_params / len(args.datasets_names)}')
        else:
            prompt_size = model.get_prompt_real_size()
            trainable_params = prompt_size * 768
            print(f'All trainable parameters: {trainable_params}; per task: {trainable_params / len(args.datasets_names)}')


    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    if is_master:
        print()
        print('***** running training *****')
        print(f'| batch_size: {args.train_batch_size} | num_epochs: {args.n_epochs} | num_train: {len(TrainDataloader)} |')

    global_step = 0
    best_dev_epoch = 0
    best_dev_step = 0
    best_epoch_dev = float('-inf')
    best_step_dev = float('-inf')
    val_res, test_res = 0, 0
    final_test = 0
    finetuned_checkpoint = None
    # try:
    while True:
        for epoch in range(int(args.n_epochs)):
            model.train()
            step_count = 0

            if epoch != 0 and len(args.datasets_names) > 1:
                training_set.reset(epoch)

            if is_mp:
                sampler.set_epoch(epoch)

            with tqdm(total=len(TrainDataloader), desc=f'Epoch {epoch}/{args.n_epochs}', unit='b', disable=args.close_tqdm) as pbar:
                update_stride = len(TrainDataloader) // 100 if len(TrainDataloader) > 200 else 1

                for step, batch in enumerate(TrainDataloader):
                    global_step += 1

                    if global_step <= resume_steps:

                        if step % update_stride == 0:
                            pbar.update(update_stride)                        
                        continue
                    
                    if len(batch['source_ids'].shape) == 3:
                        source_ids = batch['source_ids'].squeeze(0).to(local_rank)
                        source_mask = batch['source_mask'].squeeze(0).to(local_rank)
                        labels = batch['target_ids'].squeeze(0).to(local_rank)
                        task_ids = torch.tensor([x[0] for x in batch['task_ids']]).to(local_rank)
                    else:
                        source_ids = batch['source_ids'].to(local_rank)
                        source_mask = batch['source_mask'].to(local_rank)
                        labels = batch['target_ids'].to(local_rank)
                        task_ids = batch['task_ids'].to(local_rank)

                    outputs = model(input_ids=source_ids, attention_mask=source_mask, labels=labels, task_ids=task_ids)
                    loss = outputs['loss']

                    loss = loss / args.accumulate_steps
                    loss.backward()
                    step_count += 1

                    if step_count == args.accumulate_steps:

                        if args.max_grad_norm > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        step_count = 0
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    if step % update_stride == 0:
                        pbar.set_postfix(**{'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
                        pbar.update(update_stride)

                    if is_master and args.save_ckpt_dir:  # save the prompts
                        if global_step % args.saving_steps == 0:
                            checkpoint = {
                                'global_step': global_step,
                                'params': {n: p for n, p in model.named_parameters() if 'prefix' in n},
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                            }
                            file_name = '.'.join(args.datasets) + '.soft_prompts.step.{}.pt'.format(global_step)
                            save_checkpoint(args.save_ckpt_dir, checkpoint, file_name)
                            print(f"Saved soft prompts at: {os.path.join(args.save_ckpt_dir, file_name)}")

                            # Track the latest checkpoint for resuming
                            save_checkpoint(args.save_ckpt_dir, checkpoint, args.latest_ckpt_name)

                            if args.eval_in_train:
                                if "wsc" in args.datasets_names[0] or "WSC" in args.datasets_names[0]:
                                    res = task_evaluation_wsc(args, ValDataloaders, model, tokenizer, device)
                                else:
                                    res = task_evaluation(args, ValDataloaders, model, tokenizer, device)
                                if len(res) > 1:
                                    val_res = average_multi_task(res)
                                else:
                                    val_res = res[args.datasets_names[0]][TASK_TO_METRICS[args.datasets_names[0]][0]]

                                if val_res > best_step_dev:
                                    best_step_dev = val_res
                                    best_dev_step = global_step
                                    print(f'Step Best Val: {best_step_dev} at Step {global_step}: {res}')

            # Epoch saving dring the training
            if is_master and args.saving_each_epoch and global_step > resume_steps: # save prompts at the end of every epoch
                checkpoint = {
                    'global_step': global_step,
                    'params': {n: p for n, p in model.named_parameters() if 'prefix' in n},
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                file_name = '.'.join(args.datasets) + '.soft_prompts.epoch.{}.pt'.format(epoch)
                save_checkpoint(args.save_ckpt_dir, checkpoint, file_name)
                print(f"Saved soft prompts at: {os.path.join(args.save_ckpt_dir, file_name)}")

            # Epoch evaluation dring the training
            if is_master and args.eval_in_train and global_step > resume_steps:
                output_path = os.path.join(args.model_output_path, f"dev_ep_{epoch}")
                if "wsc" in args.datasets_names[0] or "WSC" in args.datasets_names[0]:
                    res = task_evaluation_wsc(args, ValDataloaders, model, tokenizer, device, output_path)
                else:
                    res = task_evaluation(args, ValDataloaders, model, tokenizer, device, output_path)
                print(f'Epoch {epoch} - Validation: ', res)
                if len(res) > 1:
                    val_res = average_multi_task(res)
                else:
                    val_res = res[args.datasets_names[0]][TASK_TO_METRICS[args.datasets_names[0]][0]]


                if val_res > best_epoch_dev:
                    best_epoch_dev = val_res
                    
                    best_dev_epoch = epoch
                    print(f'Epoch Best Val: {best_epoch_dev} at Epoch {best_dev_epoch}.')

        if is_master:
            print('***** training ends *****')
            print()
            print('best dev acc: {:.5f} (at epoch {})'.format(best_epoch_dev, best_dev_epoch))
            print('best dev acc: {:.5f} (at step {})'.format(best_step_dev, best_dev_step))
            print()

        exit()

    return


def task_evaluation_wsc(args, dataloader_dict, model, tokenizer, device, output_path=None):
        
    model.eval()
    results = {}  # tasks: {metrics}

    wsc_acc = []
    tag_labels = []
    with torch.no_grad():
        for dataset_name, data_loader in dataloader_dict.items():
            results[dataset_name] = {}
            raw_preds = []
            task_preds = []
            task_labels = []
            for batch in data_loader:
                source_ids = batch['source_ids'].to(device)
                source_mask = batch['source_mask'].to(device)
                task_ids = batch['task_ids'].to(device)
                labels = batch['target_ids']
                raw_input = batch['raw_target']
                tag_labels += [i for i in batch["label"].numpy().tolist()]

                try:
                    preds = model.generate(
                        input_ids=source_ids,
                        attention_mask=source_mask, 
                        max_length=args.max_target_length,
                        num_beams=1,
                        task_ids=task_ids,  # model_kwargs
                        ).cpu().detach()
                except:
                    preds = model.module.generate(
                        input_ids=source_ids,
                        attention_mask=source_mask, 
                        max_length=args.max_target_length,
                        num_beams=1,
                        task_ids=task_ids,  # model_kwargs
                        ).cpu().detach()

                raw_preds += preds
                batch_extra_fields = [x[0] for x in batch['extra_fields']] if isinstance(batch['extra_fields'][0], tuple) else batch['extra_fields']
                data_info = [eval(x) for x in batch_extra_fields]
                post_processor = AutoPostProcessor.get(dataset_name, tokenizer, ignore_pad_token_for_loss=True)
                decoded_preds, decoded_labels = post_processor.process(preds, labels, data_info)
                # print(decoded_preds)
                # print(decoded_labels)
                
                task_preds += decoded_preds
                task_labels += decoded_labels

                #return {"accuracy": 100 * ((np.array(predictions) == np.array(targets)).mean())}
                
                for pred, truth in zip(task_preds, raw_input):
                    flag = wsc_simple(pred, truth)
                    wsc_acc.append(flag)

            cnt = 0
            for i,j in zip(tag_labels, wsc_acc):
                if i == j:
                    cnt += 1

            for i, metric in enumerate(AutoTask.get(dataset_name).metric):
                results[dataset_name].update({"accuracy": 100*cnt/len(wsc_acc)})   

    model.train()
    return results


def task_evaluation(args, dataloader_dict, model, tokenizer, device, output_path=None):
    model.eval()
    results = {}  # tasks: {metrics}
    with torch.no_grad():
        for dataset_name, data_loader in dataloader_dict.items():
            results[dataset_name] = {}
            raw_preds = []
            task_preds = []
            task_labels = []
            for batch in data_loader:
                source_ids = batch['source_ids'].to(device)
                source_mask = batch['source_mask'].to(device)
                task_ids = batch['task_ids'].to(device)
                labels = batch['target_ids']

                try:
                    
                    preds = model.generate(
                        input_ids=source_ids,
                        attention_mask=source_mask, 
                        max_length=args.max_target_length,
                        num_beams=1,
                        task_ids=task_ids,  # model_kwargs
                        ).cpu().detach()
                except:
                    
                    preds = model.module.generate(
                        input_ids=source_ids,
                        attention_mask=source_mask, 
                        max_length=args.max_target_length,
                        num_beams=1,
                        task_ids=task_ids,  # model_kwargs
                        ).cpu().detach()

                raw_preds += preds
                batch_extra_fields = [x[0] for x in batch['extra_fields']] if isinstance(batch['extra_fields'][0], tuple) else batch['extra_fields']
                data_info = [eval(x) for x in batch_extra_fields]
                post_processor = AutoPostProcessor.get(dataset_name, tokenizer, ignore_pad_token_for_loss=True)
                decoded_preds, decoded_labels = post_processor.process(preds, labels, data_info)

                task_preds += decoded_preds
                task_labels += decoded_labels

            # store the results to files
            if output_path:
                file_path = output_path + f'_{dataset_name}.output'
                with open(file_path, 'w') as f:
                    for i in range(len(raw_preds)):
                        f.write('raw output:\t' + str(raw_preds[i]) + '\n')
                        f.write('predicted label:\t' + str(task_preds[i]) + '\n')
                        f.write('golden label:\t' + str(task_labels[i]) + '\n')
                        f.write('\n')

            for i, metric in enumerate(AutoTask.get(dataset_name).metric):
                results[dataset_name].update(metric(task_preds, task_labels))

    model.train()
    return results


def main():
    print('Stating time: ', datetime.now().strftime("%m/%d/%Y %X"))
    args = process_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Use all gpus unless gpus ids are specified 
    args.gupids = args.gupids if args.gupids else list(range(torch.cuda.device_count()))

    if len(args.gupids) > 1: 

        os.environ['MASTER_ADDR'] = 'localhost'

        while is_port_in_use(args.port):
            args.port += 1
        
        os.environ['MASTER_PORT'] = f'{args.port}'
        print("Use port", args.port)

        print("Use gpus ", args.gupids)
        args.world_size = len(args.gupids)
        mp.spawn(run, nprocs=len(args.gupids), args=(args,), join=True)

    else:
        args.world_size = 1
        print("Use single gpu!")	
        run(0, args)
    
    print("Ending time: ", datetime.now().strftime("%m/%d/%Y %X"))
    return


if __name__ == '__main__':
    main()