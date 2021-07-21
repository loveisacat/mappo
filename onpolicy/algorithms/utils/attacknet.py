import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .transformer import Transformer
import random
from .distributions import Bernoulli, Categorical, DiagGaussian

#This is the Attack action Representation Network(AttackNet)
#We encode the attack action with 7-th bit of action
#If the attaction is 7-th, we learn the actual attack enemy id.
class AttackNet(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, agent_num, device):
        super(AttackNet, self).__init__()
        action_dim = action_space.n
        self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
 
        self.device = device


    def forward(self, x, available_actions=None, deterministic=False):
        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample() 
        action_log_probs = action_logits.log_probs(actions)
        
        return actions, action_log_probs


    def transback(self, inputs):
        for i in range (0,len(inputs)):
            for j in range (0,len(inputs[i])):
                if inputs[i][j] >= 6:
                    inputs[i][j] = 6
        return inputs

    def evaluate_attacks(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_logits = self.action_out(x, available_actions)
        action_log_probs = action_logits.log_probs(action)
        if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
        else:
                dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy

if __name__ == '__main__':
    pass
