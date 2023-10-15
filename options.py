import os
import sys
import argparse

from utils import *


def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]


def process_args():

    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('-ckpt', '--from_checkpoint', default='', type=str, help='load the model from a checkpoint to resume the training')
    parser.add_argument('--auto_resume', type='bool', default=False)
    parser.add_argument('--save_ckpt_dir', type=str, default='./saved_models', help="load the latest checkpoint in this dir for resuming")

    # parser.add_argument('--save_source_dir', default='', help='model output directory')
    parser.add_argument('--load_source_path', default='', help='source prompt file for target-FT')
    parser.add_argument('--saving_steps', type=int, default=1000)
    parser.add_argument('--saving_each_epoch', type='bool', default=True)
    parser.add_argument('--latest_ckpt_name', type=str, default='latest_checkpoint.pt', help="the name of the latest checkpoint in this dir for resuming")

    # optimization
    parser.add_argument('-e', '--n_epochs', default=20, type=int, help='total number of training epochs to perform.')
    parser.add_argument('-ts', '--train_batch_size', default=32, type=int)  
    parser.add_argument('-es', "--eval_batch_size", default=128, type=int)
    parser.add_argument('--lr', default=0.3, type=float, help='learning rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='l2 weight decay strength')
    parser.add_argument('--accumulate_steps', type=int, default=1)
    
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--num_warmup_steps', type=float, default=0.1)
    parser.add_argument('--max_train_steps', type=int, default=0)

    parser.add_argument('--min_training_epoch', type=int, default=20)
    parser.add_argument('--early_stopping_patience', type=int, default=10)

    # useful arguments
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred'], help='run training or evaluation')
    
    # training arguments
    parser.add_argument('-c', '--close_tqdm', type='bool', default=True)
    parser.add_argument('-me', '--max_epochs_before_stop', default=2, type=int, help='stop training if dev does not increase for N epochs')

    parser.add_argument('--model_name', default='t5-base', help='model name')
    parser.add_argument('--datasets', type=str, default='rte', help='A list of datasets, seperated by the semicolon')

    parser.add_argument('--max_source_length', type=int, default=256)
    parser.add_argument('--max_target_length', type=int, default=128, help='SQuAD needs long decoding length')

    parser.add_argument('-pt', '--prompt_type', type=str, default='param', help='two implementations of prompting, embed or param')
    parser.add_argument('--is_dynamic_share', type='bool', default=False)
    parser.add_argument('--eval_in_train', type='bool', default=True, help='do evaluation during the training')

    # PT parameters
    parser.add_argument('-el', '--enc_prompt_tokens', type=int, default=100)
    parser.add_argument('--dec_prompt_tokens', type=int, default=0)
    parser.add_argument('-sr', '--sharing_ratio', type=float, default=1.0, help='Sharing prompts across tasks')

    # Data preprocessing
    parser.add_argument('-ss', '--share_seed', type=int, default=42)
    parser.add_argument('--add_task_prefix', type='bool', default=True, help='add task perfix by default')
    parser.add_argument('-vb', '--add_verbalizer', type='bool', default=False, help='add verbalizer to the context')

    # DDP
    # parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.') 
    parser.add_argument("-p", "--port", type=int, default=12355, help="port")
    parser.add_argument('-gp', "--gupids", nargs='+', type=int, help="default is None")

    parser.add_argument('-sm', "--sampling_method", type=str, default='uniform', choices=['uniform', 'stratified', 't5', 'unifiedqa'])
    parser.add_argument('-sl', "--size_limit", type=int, default=262144, help="size limit for t5 sampling, default is 2**18")
    parser.add_argument('-st', '--stoch_task', type='bool', default=False)


    # decomposion
    parser.add_argument('-bn', "--bottle_neck", type=int, default=10)

    parser.add_argument('--model_output_path', type=str, default='./saved_outputs', help='path to the model outputs')
    parser.add_argument('-lp', '--local_file_prefix', type=str, default='/gpfs/u/home/DPTV/DPTVhnwz/scratch/mrqa_datasets/datasets', 
                        help='local path prefix for external datasets, exp., MRQA')

    args = parser.parse_args()

    
    args.datasets = args.datasets.strip().split(',')

    args.datasets_names = [TASK_NAME_MAPPING[x] for x in args.datasets if x != '']

    if args.save_ckpt_dir and not os.path.exists(args.save_ckpt_dir):
            os.makedirs(args.save_ckpt_dir)

    if args.model_output_path and not os.path.exists(args.model_output_path):
            os.makedirs(args.model_output_path)

    if args.model_name == '':
        args.model_name = 't5-base'  # default model name

    print('Raw Arguments: ', args)
    print('Process ID: ', os.getpid())

    return args

