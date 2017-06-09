import torch
import numpy as np


class CharRNN(torch.nn.Module):

	def __init__(self, hidden_size, x_size, T = 25):
		
		super(CharRNN,self).__init__()
		self.hidden_size = hidden_size
		self.x_size      = x_size
		self.T           = T
		self.num_layers  = 1
		self.gru  = torch.nn.RNNCell(input_size = self.x_size, hidden_size = hidden_size, bias = True)
		self.linear = torch.nn.Linear(hidden_size, x_size)
		self.init_weights()

	def init_weights(self):
		self.linear.bias.data.fill_(0)
		self.linear.weight.data.uniform_(-0.1,0.1)

	def forward(self,input, h_0 = None):

		# simple forward pass
		# assuming input is a torch tensor		
		# input dimensions demanded by LSTMCell is batch_size,input_size ==> 1,input_size
		# if input == self.T : print "initialised timestep and given timestep different"

		h_t = h_0# torch.autograd.Variable(torch.zeros(1,self.hidden_size))

		y_t = []
		x_t = None


		for timestep in range(len(input)):

			x_t = input[timestep].view(1,self.x_size)
			h_t = self.gru(x_t, h_t)
			y = torch.nn.functional.softmax(self.linear(h_t))
			y_t.append(y)

		return y_t, h_t 

	def sample(self, seed_x, h_0):

		ixes = []
		x_t = seed_x
		h_t = h_0
		for timestep in range(self.T):

			h_t = self.gru(x_t,h_t)
			p_t = torch.nn.functional.softmax(self.linear(h_t))
			ix = np.random.choice(range(self.x_size), p = p_t.data.numpy().ravel())
			# ix = np.argmax(p_t.data.numpy())
			x_t = torch.autograd.Variable(torch.zeros(x_t.size()))
			x_t[0,ix] = 1
			ixes.append(ix)

		return ixes 