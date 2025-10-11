"""
Focal Loss Wrapper.  (c) 2021 Georgia Tech

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

import torch
import torch.nn as nn
import torch.nn.functional as F

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from focal_loss.py!")


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return: tensor containing the weights for each class
    """

    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    per_cls_weights = None
    n = torch.tensor(cls_num_list, dtype=torch.float32)
    beta = float(beta)
    zero = n <= 0
    effective = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), n)
    effective = torch.where(zero, torch.ones_like(effective), effective)
    w = (1.0 - beta) / effective
    w = torch.where(zero, torch.zeros_like(w), w)
    mean = w[~zero].mean().clamp(min=1e-8)
    per_cls_weights = (w / mean).clamp(max=100.0)  # cap to prevent blow-ups
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        log_probs = F.log_softmax(input, dim=1)
        log_p_t = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

        # Unweighted CE term: -log p_t
        ce = -log_p_t

        # p_t for focal modulation (NO in-place ops)
        p_t = torch.exp(log_p_t)
        p_t = torch.clamp(p_t, min=1e-6, max=1.0 - 1e-6)
        focal = torch.pow(1.0 - p_t, self.gamma)

        # class-balanced weight alpha_t (NO in-place ops)
        if self.weight is not None:
            w = self.weight.to(input.device, dtype=input.dtype)
            w = torch.clamp(w, max=20.0)
            alpha_t = w[target]
        else:
            alpha_t = torch.ones_like(ce)

        loss = (alpha_t * focal * ce).mean()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
