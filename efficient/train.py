import torch
import argparse
import os

from helpers import *
from model import *
from generate import *


argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--epochs', type=int, default=1000)
argparser.add_argument('--printtime', type=int, default=5)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=1)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--seq_length', type=int, default=100)
argparser.add_argument('--batch_size', type=int, default=10)
argparser.add_argument('--cuda', default=False, action='store_true')
args = argparser.parse_args()


file, file_len = read_file(args.filename)

def train_set(seq_length, batch_size):

    input = torch.LongTensor(batch_size, seq_length)
    target = torch.LongTensor(batch_size, seq_length)

    for i in range(batch_size):
     
        start_idx = random.randint(0, file_len - seq_length)
        end_idx = start_idx + seq_length + 1
        seq = file[start_idx:end_idx]
        input[i] = char_tensor(seq[:-1])
        target[i] = char_tensor(seq[1:])


    input = torch.autograd.Variable(input)
    target = torch.autograd.Variable(target)
    if args.cuda:
        input = input.cuda()
        target = target.cuda()

    return input, target

def train(inp, target):

    hidden = charnet.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    
    charnet.zero_grad()
    loss = 0

    for c in range(args.seq_length):
        output, hidden = charnet(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    optimizer.step()

    return loss.data[0] / args.seq_length

def save_model():

    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(charnet, save_filename)
    print('saving file as %s' % save_filename)


charnet = CharRNN(n_chars, args.hidden_size, n_chars, args.model, args.n_layers)
optimizer = torch.optim.Adam(charnet.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

if args.cuda:
    charnet.cuda()

start = time.time()
all_losses = []
loss_avg = 0

try:

    for epoch in range(1, args.epochs + 1):
        loss = train(*train_set(args.seq_length, args.batch_size))
        loss_avg += loss

        if epoch % args.printtime == 0:
            print '[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch/args.epochs * 100, loss)
            print generate(charnet, 'a', 100, cuda=args.cuda), '\n'

    print("Saving...")
    save_model()

except KeyboardInterrupt:
    print("backing up...")
save_model()