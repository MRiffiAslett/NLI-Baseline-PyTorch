import torch
import os
import logging
from argparse import ArgumentParser

def parse_arguments():
    # Setup argument parser for command line arguments
    parser = ArgumentParser(description='PyTorch/torchtext NLI Baseline')
    parser.add_argument('--dataset', '-d', type=str, default='mnli')
    parser.add_argument('--model', '-m', type=str, default='bilstm')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=200)
    parser.add_argument('--dp_ratio', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--combine', type=str, default='cat')
    parser.add_argument('--results_dir', type=str, default='results')
    return validate_arguments(parser.parse_args())

def validate_arguments(args):
    # Validate and prepare the necessary folders
    ensure_directory(os.path.join(args.results_dir, args.model, args.dataset))

    # Validate epochs
    assert args.epochs >= 1, 'Number of epochs must be larger than or equal to one'

    # Validate batch size
    assert args.batch_size >= 1, 'Batch size must be larger than or equal to one'
    return args

def select_device(gpu_index):
    # Select the appropriate device (GPU or CPU)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)
        return torch.device(f'cuda:{gpu_index}')
    else:
        return torch.device('cpu')

def create_directory(path):
    # Create a directory if it does not exist
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != os.errno.EEXIST or not os.path.isdir(path):
            raise

def ensure_directory(directory_path):
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

def setup_logger(args, phase):
    # Setup a logger for training or evaluation phase
    log_file_path = f"{args.results_dir}/{args.model}/{args.dataset}/{phase}.log"
    logging.basicConfig(level=logging.INFO, filename=log_file_path, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    return logging.getLogger(phase)
