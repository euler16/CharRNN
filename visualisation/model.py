import torch
import numpy as np



class CharRNN(torch.nn.Module):

	def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1):
		super(CharRNN, self).__init__()
		self.model = model.lower()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers

		self.embed    = torch.nn.Embedding(input_size, hidden_size)
		if self.model == "gru":
			self.rnn = torch.nn.GRU(hidden_size, hidden_size, n_layers)
		elif self.model == "lstm":
			self.rnn = torch.nn.LSTM(hidden_size, hidden_size, n_layers)
		self.h2o = torch.nn.Linear(hidden_size, output_size)

	def forward(self, input, hidden):
		batch_size = input.size(0)
		encoded = self.embed(input)
		output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
		output = self.h2o(output.view(batch_size, -1))
		return output, hidden

	def forward2(self, input, hidden):
		encoded = self.embed(input.view(1, -1))
		output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
		output = self.h2o(output.view(1, -1))
		return output, hidden

	def init_hidden(self, batch_size):
		
		if self.model == "lstm":
			return (torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
					torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
		
		return torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))