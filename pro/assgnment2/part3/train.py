from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
# import time
import numpy as np

import torch
from torch.utils.data import DataLoader
# import sys
# sys.path.extend(['/home/joshua/dl/assgnment2/part3'])
from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
import torch.nn as nn
import torch.optim as optim

'''
default:
class Config:
    input_length = 10
    input_dim = 1
    num_classes = 10
    num_hidden = 128
    batch_size = 128
    learning_rate = 0.001
    train_steps = 10000
    max_norm = 10.0

config = Config
'''


def train(config):

    # Initialize the model that we are going to use
    model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden,
                       config.num_classes, config.batch_size)

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()  # fixme
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    model = model.cuda()

    correct = 0
    total = 0
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Add more code here ...
        optimizer.zero_grad()
        batch_inputs = batch_inputs[:, :, np.newaxis]
        batch_inputs = batch_inputs.cuda()
        batch_targets = batch_targets.cuda()
        hidden_seq, batch_output = model(batch_inputs)

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        loss = criterion(batch_output, batch_targets)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(batch_output.data, 1)
        # print(predicted)
        total += batch_targets.size(0)
        correct += (predicted == batch_targets).sum().item()
        accuracy = 100 * correct / total  # fixme

        # print(predicted)
        if step % 100 == 0:
            # print accuracy/loss here
            print('Accuracy of the network on the %d test palindromes: %d %%' % (total,
                                                                                 accuracy))
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            print('Accuracy of the network on the %d test palindromes: %d %%' % (total,
                                                                                 accuracy))
            break

    print('Done training.')
    return accuracy


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    # Train the model
    train(config)




