"""
LSTM model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # You will need to complete the class init function, and forward function

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method you          #
        #   should NOT transpose the weights.                                          #
        #   You also need to include correct activation functions                      #
        ################################################################################

        # i_t: input gate  i_t = σ(x_t·W_ii + b_ii + h_{t-1}·W_hi + b_hi)
        self.weight_i_t = nn.Parameter(torch.empty(input_size, hidden_size))  # W_ii
        self.bias_i_t = nn.Parameter(torch.empty(hidden_size))  # b_ii
        self.weight_h_t = nn.Parameter(torch.empty(hidden_size, hidden_size))  # W_hi
        self.bias_h_t = nn.Parameter(torch.empty(hidden_size))  # b_hi
        self.activation_i = nn.Sigmoid()

        # f_t: forget gate  f_t = σ(x_t·W_if + b_if + h_{t-1}·W_hf + b_hf)
        self.weight_f_t = nn.Parameter(torch.empty(input_size, hidden_size))  # W_if
        self.bias_f_t = nn.Parameter(torch.empty(hidden_size))  # b_if
        self.weight_h_f = nn.Parameter(torch.empty(hidden_size, hidden_size))  # W_hf
        self.bias_h_f = nn.Parameter(torch.empty(hidden_size))  # b_hf
        self.activation_f = nn.Sigmoid()

        # g_t: cell candidate  g_t = tanh(x_t·W_ig + b_ig + h_{t-1}·W_hg + b_hg)
        self.weight_g_t = nn.Parameter(torch.empty(input_size, hidden_size))  # W_ig
        self.bias_g_t = nn.Parameter(torch.empty(hidden_size))  # b_ig
        self.weight_h_g = nn.Parameter(torch.empty(hidden_size, hidden_size))  # W_hg
        self.bias_h_g = nn.Parameter(torch.empty(hidden_size))  # b_hg
        self.activation_g = nn.Tanh()

        # o_t: output gate  o_t = σ(x_t·W_io + b_io + h_{t-1}·W_ho + b_ho)
        self.weight_o_t = nn.Parameter(torch.empty(input_size, hidden_size))  # W_io
        self.bias_o_t = nn.Parameter(torch.empty(hidden_size))  # b_io
        self.weight_h_o = nn.Parameter(torch.empty(hidden_size, hidden_size))  # W_ho
        self.bias_h_o = nn.Parameter(torch.empty(hidden_size))  # b_ho
        self.activation_o = nn.Sigmoid()

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              #
        #   h_t and c_t should be initialized to zeros.                                #
        #   ----------------------------------------------------------------------     #
        #     it = σ(xt.Wii + bii + ht−1.Whi + bhi)                                    #
        #     ft = σ(xt.Wif + bif + ht−1.Whf + bhf )                                   #
        #     gt = tanh(xt.Wig + big + ht−1.Whg + bhg )                                #
        #     ot = σ(xt.Wio + bio + ht−1.Who + bho)                                    #
        #     ct = ft ⊙ ct−1 + it ⊙ gt                                                 #
        #     ht = ot ⊙ tanh(ct)                                                       #
        ################################################################################
        # h_t, c_t = None, None  #remove this line when you start implementing your code

        batch, sequence, feature = x.shape
        device = x.device
        dtype = x.dtype

        h_t = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        c_t = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)

        for t in range(sequence):
            x_t = x[:, t, :]
            i_t = self.activation_i(x_t @ self.weight_i_t + self.bias_i_t +
                                    h_t @ self.weight_h_t + self.bias_h_t)

            f_t = self.activation_f(x_t @ self.weight_f_t + self.bias_f_t +
                                    h_t @ self.weight_h_f + self.bias_h_f)

            g_t = self.activation_g(x_t @ self.weight_g_t + self.bias_g_t +
                                    h_t @ self.weight_h_g + self.bias_h_g)

            o_t = self.activation_o(x_t @ self.weight_o_t + self.bias_o_t +
                                    h_t @ self.weight_h_o + self.bias_h_o)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
