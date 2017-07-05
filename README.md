# Char RNN

This repository contains PyTorch code for training and visualising (through heatmaps) Recurrent Neural Networks.

## Recurrent Neural Networks
Recurrent Nets are Neural Network architecure for modelling sequences. The code in the repository is about the 2 main variants of RNN :- **LSTM and GRU**. <br>
The working of RNN can be summarized in the following gif
<img src="assets/rnn.gif" />

The input to the RNN in our case is vectorized representation of characters.

## Installation

Create and activate a virtualenv.<br>
-> virtualenv charrnn<br>
-> source charrnn/bin/activate<br>

Install [ __PyTorch__ ](https://pytorch.org)
(see requirements.txt for the version, listed there  as torch)
<br>
Clone the repository <br>
-> git clone https://github.com/euler16/CharRNN.git<br>
-> cd CharRNN<nr>

Install other dependencies<br>
-> pip install -r requirements.txt<br>
Note:- not all dependencies mentioned in the requirements.txt file are required.

## Running the code

The model contains 3 main folders:- efficient, simple, visualisation.
The simple folder contains code for implementing an RNN from scratch (without using the LSTM/GRU module from Pytorch). The efficient folder contains code using the modules from PyTorch and also uses embedding table. The visualisation folder contains code for generating heatmaps.
<br>
In all three repositories run<br> 
-> python train.py --path/to/data<br>
Check the argument parser code to know more about the arguments available

## Visualization

<img src="assets/bokeh_plot.png" />
In the visualisation folder, train a model using <br>
-> python vis_generate.py
Then run<br>
-> python heatmap_plot.py
Please see the argument parsers as well as the files that are being loaded and saved!! (if you are still facing problems file an issue here (code and installation related issues only)).

This code has been used for [ __this__ ](https://euler16.github.io/cs/2017/07/01/playing-with-rnn.html) blog post.