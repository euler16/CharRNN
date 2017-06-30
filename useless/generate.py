import torch
import os
import argparse

from helper import *
from model import *

def generate(charnet, char2idx, idx2char, prime_str='A', predict_len=100, temperature=0.8, cuda=False):
    h_t, c_t = charnet.init_hidden(1)
    prime_input = torch.autograd.Variable(char2tensor(prime_str,char2idx).unsqueeze(0))

    if cuda:
        h_t = h_t.cuda()
        c_t = c_t.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, h_t, c_t = charnet(prime_input[:,p], h_t, c_t)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, h_t, c_t = charnet(inp, h_t, c_t)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = idx2char[top_i]
        predicted += predicted_char
        inp = torch.autograd.Variable(char2tensor(predicted_char,char2idx).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted

# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    decoder = torch.load(args.filename)
    del args.filename
    print(generate(decoder, **vars(args)))