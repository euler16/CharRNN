import torch
import numpy as np


class StableBCELoss(torch.nn.modules.Module):
	
	def __init__(self):
		super(StableBCELoss, self).__init__()
	def forward(self, input, target):
		neg_abs = - input.abs()
		loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
		return loss.mean()


class CharRNN(torch.nn.Module):

	def __init__(self,input_size,hidden_size,output_size, n_layers = 1):

		super(CharRNN, self).__init__()
		self.input_size  = input_size
		self.hidden_size = hidden_size
		self.n_layers    = 1

		self.x2h_i = torch.nn.Linear(input_size + hidden_size, hidden_size)
		self.x2h_f = torch.nn.Linear(input_size + hidden_size, hidden_size)
		self.x2h_o = torch.nn.Linear(input_size + hidden_size, hidden_size)
		self.x2h_q = torch.nn.Linear(input_size + hidden_size, hidden_size)
		self.h2o   = torch.nn.Linear(hidden_size, output_size)
		self.sigmoid = torch.nn.Sigmoid()
		self.softmax = torch.nn.Softmax()
		self.tanh    = torch.nn.Tanh()

	def forward(self, input, h_t, c_t):

		combined_input = torch.cat((input,h_t),1)

		i_t = self.sigmoid(self.x2h_i(combined_input))
		f_t = self.sigmoid(self.x2h_f(combined_input))
		o_t = self.sigmoid(self.x2h_o(combined_input))
		q_t = self.tanh(self.x2h_q(combined_input))

		c_t_next = f_t*c_t + i_t*q_t
		h_t_next = o_t*self.tanh(c_t_next)

		output = self.h2o(h_t_next) # did not apply softmax as using BCELOSS
		return output, h_t, c_t
		
	def initHidden(self):
		return torch.autograd.Variable(torch.zeros(1, self.hidden_size))

	def weights_init(self,model):
		
		classname = model.__class__.__name__
		if classname.find('Linear') != -1:
			model.weight.data.normal_(0.0, 0.02)
			model.bias.data.fill_(0)
			