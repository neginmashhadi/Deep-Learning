"""
2d Convolution Module.  (c) 2021 Georgia Tech

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

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from convolution.py!")

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        N, C, H, W = x.shape # batch_size, color channels, height of input volume, width of input wolume
        F, Cw, KH, Kw = self.weight.shape # number of filters, number of in channels, height of filter volume,
                                          # width of filter volume

        s = self.stride
        p = self.padding

        # H_out = ((H + 2 * p - KH) // s) + 1 # height output
        # W_out = ((W + 2 * p - Kw) // s) + 1 # weight output

        H_out = 1 + (x.shape[2] + 2 * p - KH) // s
        W_out = 1 + (x.shape[3] + 2 * p - Kw) // s

        x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
        out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)

        for i in range(H_out):
            h0 = i * s
            h1 = h0 + KH
            for j in range(W_out):
                w0 = j * s
                w1 = w0 + Kw
                sliding_window = x_pad[:, None, :, h0:h1, w0:w1]
                out[:, :, i, j] = np.sum(sliding_window * self.weight[None, :, :, :, :], axis=(2, 3, 4))

        # add bias
        out += self.bias[None, :, None, None]
        x = (x_pad, s, p, KH, Kw, H_out, W_out)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        x_pad, s, p, KH, KW, H_out, W_out = x
        N, C_in, H_pad, W_pad = x_pad.shape
        F, _, _, _ = self.weight.shape  # out channels

        dx_pad = np.zeros_like(x_pad)
        dw = np.zeros_like(self.weight)
        db = np.zeros_like(self.bias)

        # db: sum dout over N and spatial
        for f in range(F):
            db[f] = np.sum(dout[:, f, :, :])

        # dw and dx
        for n in range(N):
            for i in range(H_out):
                h_start = i * s
                h_end = h_start + KH
                for j in range(W_out):
                    w_start = j * s
                    w_end = w_start + KW
                    # slice once for speed
                    x_window = x_pad[n, :, h_start:h_end, w_start:w_end]  # (C_in, KH, KW)
                    for f in range(F):
                        grad = dout[n, f, i, j]
                        dw[f] += grad * x_window  # (C_in, KH, KW)
                        dx_pad[n, :, h_start:h_end, w_start:w_end] += grad * self.weight[f]
                        # self.weight[f] is (C_in, KH, KW)

        # unpad dx
        dx = dx_pad[:, :, p:-p, p:-p] if p > 0 else dx_pad

        # write grads
        self.dx = dx
        self.dw = dw
        self.db = db

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


