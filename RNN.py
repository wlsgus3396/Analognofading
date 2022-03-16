import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
import ipdb

from tqdm import tqdm
import string
import random
import re
import time, math


from io import open
import glob
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def string_to_onehot(string, seq_len=1):
    start = np.zeros(shape=len(char_list), dtype=int)
    end = np.zeros(shape=len(char_list), dtype=int)
    start[-1] = 1  # This is initial input

    # Insert zeros for initial input
    for i in range(seq_len):
        if i == 0:
            onehot = start
        else:
            onehot = np.vstack([start, onehot])

    # Convert string to one-hot vector
    for i in string:
        idx = char_list.index(i)
        zero = np.zeros(shape=char_len, dtype=int)
        zero[idx] = 1
        onehot = np.vstack([onehot, zero])

    return onehot


# Onehot vector to word
# [1 0 0 ... 0 0] -> a

def onehot_to_word(onehot_1):
    out = ''
    for i in range(onehot_1.size(0)):
        onehot = torch.Tensor.numpy(onehot_1[i, :])
        out = out + char_list[onehot.argmax()]
    return out


# Recall the target problem

string = "hello pytorch. how long can a rnn cell remember? show me your limit!"
chars = "abcdefghijklmnopqrstuvwxyz ?!.,:;0"
char_list = [i for i in chars]
char_len = len(char_list)
input_size = char_len
hidden_size = 50
output_size = char_len

# Additional settings
batch_size = 1
num_layers = 2
seq_len = 3 # In this time, we use seq_len as 3.


lr = 0.01
num_epochs = 400


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # Use LSTM model in PyTorch
        self.h2o = nn.Linear(hidden_size, output_size)  # Adjust to output size utilizing linear layer

    def forward(self, input, hidden, cell):
        ## nn.LSTM Input data shape: (seq_len, batch_size, input_size)
        ## nn.LSTM Output data shape: (seq_len, batch_size, hidden_size)

        output_LSTM, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = self.h2o(output_LSTM[-1, :, :]).reshape(batch_size, output_size)  # Get last hidden state

        return output, hidden, cell

    def init_hidden_cell(self):
        hidden = torch.zeros(num_layers, batch_size, hidden_size)
        cell = torch.zeros(num_layers, batch_size, hidden_size)

        return hidden, cell


lstm = LSTM(input_size,hidden_size,output_size,num_layers)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)



one_hot = torch.from_numpy(string_to_onehot(string,seq_len)).type_as(torch.FloatTensor()) #Convert the given sentence to one-hot vector

# Input of LSTM during training # Note that 3 zeros are inserted (since sequence length is 3)
print(onehot_to_word(one_hot))

for i in range(num_epochs):
    lstm.zero_grad()
    total_loss = 0
    hidden, cell = lstm.init_hidden_cell()

    for j in range(len(string)):
        ###################################### Write your code here ##########################################

        input_data = one_hot[j:j + seq_len].view(seq_len, batch_size, -1)  # Load x_(t), x_(t+1), x_(t+2) and change the shape of input_data to (sequence length, batch size, input size)
        output, hidden, cell = lstm(input_data, hidden, cell)  # Put the input and hidden state and cell state as input of lstm
        target = one_hot[j + seq_len].view(1,-1)  # Load groundtruth of y_(t+2) from one-hot and change the shape to (batch size, output size)
        loss = loss_func(output, target)  # Compute loss at each time step

        ######################################################################################################

        total_loss += loss

    total_loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print('At {:3d}th epoch, Loss : {:0.4f}'.format(i, total_loss.item()))


hidden,cell = lstm.init_hidden_cell()
input_data = one_hot[0:0+seq_len].view(seq_len,1,-1)

for j in range(len(string)):
    output, hidden, cell = lstm(input_data,hidden,cell)
    print(onehot_to_word(output.data),end="")
    input_data = torch.cat((input_data[1:,:,:],output.data.view(1,1,-1)),dim=0)


ipdb.set_trace()