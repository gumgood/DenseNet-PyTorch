import argparse
import torch

_parser = argparse.ArgumentParser()

# Dataset options
_parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar10+', 'cifar100', 'cifar100+', 'svhn'])

# Model options
_parser.add_argument('--bc', action='store_true')
_parser.add_argument('--num-layer', type=int, default=40)
_parser.add_argument('--growth-rate', type=int, default=12)

# Training options
_parser.add_argument('--batch-size', type=int, default=64)
_parser.add_argument('--epochs', type=int, default=300)
_parser.add_argument('--start-epoch', type=int, default=0)
_parser.add_argument('--print-freq', type=int, default=100)
_parser.add_argument('--save-freq', type=int, default=10)

# Optimization options
_parser.add_argument('--lr', type=float, default=0.1)
_parser.add_argument('--momentum', type=float, default=0.9)
_parser.add_argument('--weight-decay', type=float, default=1e-4)

args = _parser.parse_args()

# PyTorch configuration
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
