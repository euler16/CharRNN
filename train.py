import torch
import numpy as np

from matplotlib.pyplot import plt

from model import CharRNN

FILENAME = "/home/Documents/summers2017/Research/DeepLearning/"

data = open(FILENAME, 'r').read()
chars = list(set(data))
# each character will be represented as sparse one-hot vectors
data_size, x_size = len(data), len(chars)  
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

charnet   = CharRNN(hidden_size,x_size, seq_length)
loss_fn   = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(charnet.parameters(), lr = learning_rate)


