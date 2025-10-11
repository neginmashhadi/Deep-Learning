"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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
    print("Roger that from max_pool.py!")

class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N, C, H, W = x.shape
        H_pool = self.kernel_size
        W_pool = self.kernel_size
        s = self.stride

        H_out = ((H - H_pool) // s) + 1
        W_out = ((W - W_pool) // s) + 1
        out = np.empty((N, C, H_out, W_out), dtype=x.dtype)

        for i in range(H_out):
            h_start = i * s
            h_end = h_start + H_pool
            for j in range(W_out):
                w_start = j * s
                w_end = w_start + W_pool
                sliding_window = x[:, :, h_start:h_end, w_start:w_end]  # (N, C, H_pool, W_pool)
                out[:, :, i, j] = np.max(sliding_window, axis=(2, 3))   # (N, C)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        H_pool = self.kernel_size
        W_pool = self.kernel_size
        s = self.stride

        dx = np.zeros_like(x)

        for i in range(H_out):
            h_start = i * s
            h_end = h_start + H_pool
            for j in range(W_out):
                w_start = j * s
                w_end = w_start + W_pool

                sliding_window = x[:, :, h_start:h_end, w_start:w_end]          # (N, C, H_pool, W_pool)
                max_vals = np.max(sliding_window, axis=(2, 3), keepdims=True)   # (N, C, 1, 1)
                mask = (sliding_window == max_vals).astype(x.dtype)  #           (N, C, H_pool, W_pool)

                dx[:, :, h_start:h_end, w_start:w_end] += mask * dout[:, :, i, j][:, :, None, None]

        self.dx = dx
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
