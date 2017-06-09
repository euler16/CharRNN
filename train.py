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
loss_fn   = torch.NLLLoss()
optimizer = torch.optim.Adam(charnet.parameters(), lr = learning_rate)

loss_data = []

for i in  range(1):

	hprev = torch.autograd.Variable(torch.zeros(1,hidden_size), requires_grad = False)
	cprev = torch.autograd.Variable(torch.zeros(1,hidden_size), requires_grad = False)
	window_idx = 0
	iteration = 1

	input_tensor = torch.autograd.Variable(torch.zeros(seq_length,x_size),requires_grad = False)
	# output_tensor = torch.autograd.Variable(torch.zeros(seq_length,x_size),requires_grad = False)
	net_tensor = torch.autograd.Variable(torch.zeros(output_tensor.size()),requires_grad = False)

	while window_idx + seq_length + 1 >= len(data):


		inputs  = [char_to_ix[ch] for ch in data[window_idx:window_idx+seq_length]]
		targets = [char_to_ix[ch] for ch in data[window_idx+1:window_idx+seq_length+1]]

	
		for charac in range(seq_length):
			input_tensor[charac,inputs[charac]] = 1
			# output_tensor[charac,targets[charac]] = 1

		if iteration%100 == 0:
			sample_ix = charnet.sample(input_tensor[0].view(1,x_size), hprev, cprev)
			txt = ''.join(ix_to_char[ix] for ix in sample_ix)
			print '----\n %s \n----' % (txt, )

		y_t, hprev, cprev = charnet(input_tensor,hprev,cprev)
		torch.cat(y_t,net_tensor)

		loss = loss_fn(net_tensor, torch.LongTensor(targets))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		loss_data

		input_tensor.data.fill_(0)
		output_tensor.data.fill_(0)

		window_idx += seq_length
		iteration += 1



