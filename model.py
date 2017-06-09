import torch
import numpy as np

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
		self.h2o   = torch.nn.Linear(hidden_size, ouput_size)

		self.dropout = torch.nn.Dropout(0.1)
		self.softmax = torch.nn.LogSoftmax()

	def forward(self,category, input, h_t, c_t):

		combined_input = torch.cat((input,h_t),1)
		i_t = torch.nn.Sigmoid(self.x2h_i(combined_input))
		f_t = torch.nn.Sigmoid(self.x2h_f(combined_input))
		o_t = torch.nn.Sigmoid(self.x2h_o(combined_input))
		q_t = torch.nn.Tanh(self.x2h_q(combined_input))

		c_t_next = f_t*c_t + i_t*q_t
		h_t_next = o_t*torch.nn.Tanh(c_t_next)

		output = self.softmax(self.h2o(h_t_next))
		return output, h_t, c_t
		
	def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))