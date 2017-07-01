import random
from scipy.stats import bernoulli
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Configuration
MAX_NESTING = 4

teststring = ""

current_depth = 0
for i in xrange(1000000):
	if current_depth == 0: 
		allowed_chars = ["( ", "0 "]
	elif current_depth == MAX_NESTING:
		allowed_chars = [") ", str(current_depth) + " "]
	else:
		allowed_chars = [") ", "( ", str(current_depth) + " "]
	
	new_pick = random.choice(allowed_chars)
	if new_pick == ") ":
		current_depth -=1
	elif new_pick == "( ":
		current_depth +=1
		
	teststring += new_pick

with open("../data/paren-train.txt", 'w') as f:
	f.write(teststring)

teststring = ""
current_depth = 0
for i in xrange(100000):
	if current_depth == 0: 
		allowed_chars = ["( ", "0 "]
	elif current_depth == MAX_NESTING:
		allowed_chars = [") ", str(current_depth) + " "]
	else:
		allowed_chars = [") ", "( ", str(current_depth) + " "]
	
	new_pick = random.choice(allowed_chars)
	if new_pick == ") ":
		current_depth -=1
	elif new_pick == "( ":
		current_depth +=1
		
	teststring += new_pick
with open("../data/paren-valid.txt", 'w') as f:
	f.write(teststring)