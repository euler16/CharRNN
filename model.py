import torch
import numpy as np


class CharRNN(torch.nn.Module):

	def __init__(self, hidden_size, x_size, T = 25):
		
		super(CharRNN,self).__init__()
		self.hidden_size = hidden_size
		self.x_size      = x_size
		self.T           = T
		self.num_layers  = 1

		self.lstm  = torch.nn.LSTMCell(input_size = self.x_size, hidden_size = hidden_size, bias = True)
	

	def forward(self,input, h_0 = None, c_0 = None):

		# simple forward pass
		# assuming input is a torch tensor		
		# input dimensions demanded by LSTMCell is batch_size,input_size ==> 1,input_size
		# if input == self.T : print "initialised timestep and given timestep different"

		h_t = h_0# torch.autograd.Variable(torch.zeros(1,self.hidden_size))
		c_t = c_0# torch.autograd.Variable(torch.zeros(1,self.hidden_size))

		y_t = []
		x_t = None

		for timestep in len(input):

			x_t = input[timestep].view(1,len(1,x_size))
			h_t, c_t = self.lstm(x_t, (h_t,c_t))
			y_t.append(torch.nn.functional.softmax(h_t))

		return y_t, h_t, c_t 