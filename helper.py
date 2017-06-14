import torch
import numpy as np

import string

def seq2tensor(seq,x_size):

	# input_tensor --> seq_length X x_size
	seq_length = len(seq)
	tensor = torch.zeros(seq_length,x_size)
	for i in range(seq_length):
		tensor[i, char2idx[seq[i]]] = 1

	return tensor
