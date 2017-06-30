import torch
import numpy as np

import random
import time
import math
import os

def char2tensor(seq, char2idx):

	tensor = torch.zeros(len(seq)).long()
	for i in range(len(seq)):
		try:
			tensor[i] = char2idx[string[c]]
		except:
			continue
	return tensor


def time_since(since):
	s = time.time() - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


