import torch
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str,default="wa.txt" )
argparser.add_argument('--n_epochs', type=int, default=200)
argparser.add_argument('--print_every', type=int, default=5)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=1)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--seq_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
args = argparser.parse_args()

