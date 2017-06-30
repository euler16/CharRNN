import torch
import numpy as np

from model import *
from generate import *
from tqdm import tqdm
from helper import *


import matplotlib.pyplot as plt
import argparse
import time
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--filename', type=str,default="data/alice.txt" )
argparser.add_argument('--n_epochs', type=int, default=100)
argparser.add_argument('--print_every', type=int, default=5)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=1)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--seq_length', type=int, default=20)
argparser.add_argument('--batch_size', type=int, default=10)
argparser.add_argument('--input_size', type=int, default=50)
argparser.add_argument('--cuda', default=False, action='store_true')
args = argparser.parse_args()

data = open(args.filename, 'r').read()
corpus_size = len(data)
x_size = len(set(data))

char2idx = {ch:i for i,ch in enumerate(set(data))}
idx2char = {i:ch for i,ch in enumerate(set(data))}

hidden_size   = args.hidden_size
seq_length    = args.seq_length
output_size   = x_size                # both input and targets to be one hot tensor
learning_rate = args.learning_rate
input_size    = args.input_size       # dimensionality of char vectors
n_embedding   = x_size                # number of characters
n_layers      = args.n_layers
batch_size    = args.batch_size

def save():
	save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
	torch.save(charnet, save_filename)
	print('Saved as %s' % save_filename)



def train_set(seq_length, batch_size):
	input = torch.LongTensor(batch_size, seq_length)
	target = torch.LongTensor(batch_size, seq_length)
	for i in range(batch_size):
		start_idx = random.randint(0, corpus_size - seq_length)
		end_idx = start_idx + seq_length + 1
		seq = data[start_idx:end_idx]
		input[i]  = char2tensor(seq[:-1], char2idx)
		target[i] = char2tensor(seq[1:], char2idx)
	input = torch.autograd.Variable(input)
	target = torch.autograd.Variable(target)

	if args.cuda:
		input = input.cuda()
		target = target.cuda()


	return input, target


def train(input, target):
	h_t, c_t = charnet.init_hidden(batch_size)

	if args.cuda:
		h_t = h_t.cuda()
		c_t = c_t.cuda()

	charnet.zero_grad()
	loss = 0

	for c in range(seq_length):
		output, h_t, c_t = charnet(input[:,c], h_t, c_t)
		loss += criterion(output.view(batch_size, -1), target[:,c])

	loss.backward()
	torch.nn.utils.clip_grad_norm(charnet.parameters(), 1.1)
	optimizer.step()

	return loss.data[0] / seq_length


charnet = CharRNN(input_size, hidden_size, output_size, n_embedding, n_layers)
optimizer = torch.optim.Adam(charnet.parameters(), lr = learning_rate)

criterion = torch.nn.CrossEntropyLoss()

if args.cuda:
	decoder.cuda()


start = time.time()
all_losses = []
loss_avg = 0
losses = []

try:

	print("Training for %d epochs..." % args.n_epochs)
	for epoch in tqdm(range(1, args.n_epochs + 1)):
		loss = train(*train_set(seq_length, batch_size))
		loss_avg += loss
		losses.append(loss)

		if epoch % args.print_every == 0:
			print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
			print(generate(charnet, char2idx, idx2char, 'Wh', 100, cuda=args.cuda), '\n')

	print("Saving...")
	save()

except KeyboardInterrupt:
	print("Saving before quit...")
save()

x = np.arange(len(losses))
losses = np.array(losses)

plt.plot(x,losses)
plt.show()