import torch
import os
import argparse
import numpy as np
import pandas as pd

from helpers import *
from model import *

def vis_generate(decoder, input_str, temperature=0.8):

	hidden = decoder.init_hidden(1)
	test_len = len(input_str)
	prime_input = torch.autograd.Variable(char_tensor(input_str[0]).unsqueeze(0))

	for p in range(len(prime_input) - 1):
		_, hidden = decoder(prime_input[:,p], hidden)
	hidden_matrix = np.copy(hidden.unsqueeze(0).data.numpy())
	hidden_matrix = hidden_matrix.reshape((1,hidden_matrix.size))
	inp = prime_input[:,-1]
	for p in range(1,test_len):
		output, hidden = decoder(inp, hidden)
		hidden_matrix = np.vstack((hidden_matrix, hidden[0,0,:].data.numpy()))
		# print hidden[0,0,:].data.numpy()
		# Sample from the network as a multinomial distribution
		# output_dist = output.data.view(-1).div(temperature).exp()
		# top_i = torch.multinomial(output_dist, 1)[0]

		# predicted_char = chars[top_i]
		inp = torch.autograd.Variable(char_tensor(input_str[p]).unsqueeze(0))
	hidden_matrix = np.delete(hidden_matrix, 0, 0)
	df = pd.DataFrame(hidden_matrix)
	df.to_csv('paran-data-df.csv')
	np.savetxt("paren-data.csv", hidden_matrix, delimiter=",")
	np.savetxt("paren-data.tsv", hidden_matrix, delimiter="\t")

# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
	argparser = argparse.ArgumentParser()
	argparser.add_argument('filename', type=str)
	argparser.add_argument('--data_size',type=str,default=100)
	args = argparser.parse_args()

	decoder = torch.load(args.filename)
	del args.filename

	file = open("../data/paren-train.txt",'r').read()
	data = file[:args.data_size]
	vis_generate(decoder, data)
