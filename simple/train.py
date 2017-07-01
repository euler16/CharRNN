import torch
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os

from model import *
# from helper import *

argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--n_epochs', type=int, default=1500)
argparser.add_argument('--print_every', type=int, default=5)
argparser.add_argument('--hidden_size', type=int, default=600)
argparser.add_argument('--n_layers', type=int, default=1)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--seq_length', type=int, default=20)
argparser.add_argument('--batch_size', type=int, default=100)
args = argparser.parse_args()

def seq2tensor(seq,x_size):

	# input_tensor --> seq_length X x_size
	seq_length = len(seq)
	tensor = torch.zeros(seq_length,x_size)
	for i in range(seq_length):
		tensor[i, char2idx[seq[i]]] = 1

	return tensor

def save_model():

    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(charnet, save_filename)
    print('saving file as %s' % save_filename)


data = open(args.filename, 'r').read()
corpus_size = len(data)
x_size = len(set(data))  # vocab size as input dimensionality


# required by the helper functions
char2idx = {ch:i for i,ch in enumerate(set(data))}
idx2char = {i:ch for i,ch in enumerate(set(data))}

hidden_size   = args.hidden_size
seq_length    = args.seq_length
output_size   = x_size # both input and targets to be one hot tensor
learning_rate = args.learning_rate

input_tensor  = torch.autograd.Variable(torch.zeros(seq_length,x_size))
target_tensor = torch.autograd.Variable(torch.zeros(seq_length,x_size))

charnet   = CharRNN(input_size = x_size, hidden_size = hidden_size, output_size = output_size)
charnet.apply(charnet.weights_init)
criterion = StableBCELoss()
optimizer = torch.optim.Adam(charnet.parameters(), lr = learning_rate)

for i in range(args.n_epochs):

	if i%10 == 0:
		print "Iteration", i
	start_idx    = np.random.randint(0, corpus_size-seq_length-1)
	train_data   = data[start_idx:start_idx + seq_length + 1]
	input_tensor = torch.autograd.Variable(seq2tensor(train_data[:-1],x_size), requires_grad = True)
	target_tensor= torch.autograd.Variable(seq2tensor(train_data[1:],x_size), requires_grad = False)
	# print input_tensor.size(), input_tensor.data.numpy()

	# print [char2idx[ch] for ch in train_data[1:]]
	# print train_data[1:]

	# target = torch.autograd.Variable(torch.LongTensor([char2idx[ch] for ch in train_data[1:]]))

	loss = 0
	h_t = torch.autograd.Variable(torch.zeros(1,hidden_size))
	c_t = torch.autograd.Variable(torch.zeros(1,hidden_size))

	for timestep in range(seq_length):

		output, h_t, c_t = charnet(input_tensor[timestep].view(1,x_size), h_t, c_t)
		# print target_tensor[timestep].size(), x_size
		loss += criterion(output,target_tensor[timestep].view(1,x_size))

	# print "loss", loss.data.numpy()

	loss.backward()
	optimizer.step()
	optimizer.zero_grad()

	if i%20 == 0:

		x_t = input_tensor[0].view(1,x_size)
		# print "x:-", x_t
		h_t = torch.autograd.Variable(torch.zeros(1,hidden_size))
		c_t = torch.autograd.Variable(torch.zeros(1,hidden_size))

		gen_seq = []
		for timestep in range(100):
			output, h_t, c_t = charnet(x_t, h_t, c_t)
			output = charnet.softmax(output)
			ix = np.random.choice(range(x_size), p=output.data.numpy().ravel())
			x_t = torch.autograd.Variable(torch.zeros(1,x_size))
			x_t[0,ix] = 1
			gen_seq.append(idx2char[ix])

		txt = ''.join(gen_seq)
		print '----------------------'
		print txt
		print '----------------------'

save_model()