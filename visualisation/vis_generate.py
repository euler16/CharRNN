import torch
import os
import argparse

from helpers import *
from model import *

def vis_generate(decoder, input_str='ABCD', predict_len=100, temperature=0.8, cuda=False):

	hidden = decoder.init_hidden(1)
	test_len = len(input_str)
	print test_len
	prime_input = torch.autograd.Variable(char_tensor(input_str[0]).unsqueeze(0))

	for p in range(len(prime_input) - 1):
		_, hidden = decoder(prime_input[:,p], hidden)
		
	inp = prime_input[:,-1]
	for p in range(1,test_len):
		print input_str[p]
		output, hidden = decoder(inp, hidden)
		print hidden
		# Sample from the network as a multinomial distribution
		# output_dist = output.data.view(-1).div(temperature).exp()
		# top_i = torch.multinomial(output_dist, 1)[0]

		# predicted_char = chars[top_i]
		inp = torch.autograd.Variable(char_tensor(input_str[p]).unsqueeze(0))


# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
	argparser = argparse.ArgumentParser()
	argparser.add_argument('filename', type=str)
	argparser.add_argument('-p', '--input_str', type=str, default='ABCD')
	argparser.add_argument('-l', '--predict_len', type=int, default=100)
	argparser.add_argument('-t', '--temperature', type=float, default=0.8)
	argparser.add_argument('--cuda', action='store_true')
	args = argparser.parse_args()

	decoder = torch.load(args.filename)
	del args.filename
	vis_generate(decoder, **vars(args))
