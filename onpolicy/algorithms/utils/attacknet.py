import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .transformer import Transformer
import random

#This is the Attack action Representation Network(AttackNet)
#We encode the attack action with 7-th bit of action
#If the attaction is 7-th, we learn the actual attack enemy id.
class AttackNet(nn.Module):
    def __init__(self, agent_num, device):
        super(AttackNet, self).__init__()
        self.n_actions = 6
        self.n_agent = agent_num
        self.n_enemy = agent_num
        self.n_ally = agent_num
        #heads must be 1,2,3,4 ... which can be divided by (5 + 5 + 6)
        self.heads = 1
        self.device = device

        #Init the attack representation  decoder
        #self.dyn_attention = Transformer(self.enemy_shape + self.ally_shape + self.n_actions, self.enemy_shape + self.ally_shape + self.n_actions, self.heads)
        #Init the simple NN encoder
        self.fc_net = nn.Linear(1,1)
         

    def forward(self, inputs, states):
        inputs = torch.from_numpy(inputs).float()
        states = torch.from_numpy(states).float()
        for i in range (0,len(inputs)):
            for j in range (0,len(inputs[i])):
                if inputs[i][j] == 6:
                    #output = F.relu(self.fc_net(inputs[i][j]))
                    output = F.sigmoid(self.fc_net(states))
                    if(output<=0.3333):
                        output = 0
                    elif(output<=0.6666):
                        output = 1
                    else:
                        output = 2
                    #output = random.randint(0, self.n_agent - 1)
                    inputs[i][j] = inputs[i][j] + output
        inputs = inputs.detach().numpy()

        return inputs
    def backward(self, inputs):
        for i in range (0,len(inputs)):
            for j in range (0,len(inputs[i])):
                if inputs[i][j] >= 6:
                    inputs[i][j] = 6
        return inputs


if __name__ == '__main__':
    pass
