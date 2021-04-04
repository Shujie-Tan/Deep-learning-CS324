from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import Parameter
# import numpy as np


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.W_hx = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_hh = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_h = Parameter(torch.Tensor(hidden_dim))

        self.W_ph = Parameter(torch.Tensor(hidden_dim, output_dim))
        self.b_o = Parameter(torch.Tensor(output_dim))
        self.init_weights()

    def init_weights(self):
        # print(self.parameters())
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x):
        # Implementation here ...
        """Assumes x is of shape (batch, sequence, feature)"""
        h_t = torch.zeros(self.hidden_dim).to(x.device)
        hidden_seq = []
        y_t = torch.zeros(self.output_dim)
        for t in range(self.seq_length):
            x_t = x[:, t, :]
            h_t = torch.tanh(x_t @ self.W_hx + h_t @ self.W_hh + self.b_h)
            o_t = h_t @ self.W_ph + self.b_o
            y_t = torch.sigmoid(o_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch) to (batch, sequence)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, y_t

# add more methods here if needed
