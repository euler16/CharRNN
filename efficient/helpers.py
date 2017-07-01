import unidecode
import string
import random
import time
import math
import torch


chars   = string.printable
n_chars = len(chars)

def read_file(filename):
    file = open(filename).read()
    return file, len(file)


def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = chars.index(string[c])
        except:
            continue
    return tensor

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
