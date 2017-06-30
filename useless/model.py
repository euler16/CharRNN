import torch
import numpy as np



class CharRNN(torch.nn.Module):
	
	def __init__(self,input_size, hidden_size,output_size, n_embedding, n_layers = 1):

		super(CharRNN, self).__init__()
		
		self.input_size = input_size
		self.n_embedding  = n_embedding
		self.hidden_size = hidden_size
		self.n_layers    = 1

		self.embed = torch.nn.Embedding(n_embedding, input_size)  # 

		self.rnn = torch.nn.LSTM(
									input_size, 
									hidden_size,
									n_layers,
									bias = True, 
									dropout = 0.1
								)
		self.h2o = torch.nn.Linear(hidden_size, output_size)



	def forward(self, input, h_0, c_0):

		# input is a tensor of indices
		# h_t = n_layers, batch, hidden_size 
		# c_t = n_layers, batch, hidden_size

		batch_size = input.size(0)
		input_vec  = self.embed(input)
		output, (h_t, c_t) = self.rnn(input_vec.view(1,batch_size,-1),(h_0,c_0))
		output = self.h2o(h_t[-1])

		return output,h_t,c_t


		
	def init_hidden(self, batch_size):
		
		return (torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
				torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
	
	def weights_init(self,model):
		
		classname = model.__class__.__name__
		if classname.find('Linear') != -1:
			model.weight.data.normal_(0.0, 0.02)
			model.bias.data.fill_(0)
