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
        self.aset = set()
        self.bset = set()

        #Init the attack representation  decoder
        #self.dyn_attention = Transformer(self.enemy_shape + self.ally_shape + self.n_actions, self.enemy_shape + self.ally_shape + self.n_actions, self.heads)
        #Init the simple NN encoder
        self.fc_net = nn.Linear(60,1)
         

    def forward(self, inputs, states):
        inputs = torch.from_numpy(inputs).float()
        states = torch.from_numpy(states).float()
        step = 111
        gap = s_min = s_max = 0
        for i in range (0,len(inputs)):
            for j in range (0,len(inputs[i])):
                if inputs[i][j] == 6:
                    if step < 0:
                      output = random.randint(0, self.n_agent - 1)
                    else:
                        s_flat = torch.flatten(states)
                        if(len(s_flat) > 60):
                          s_reshape = s_flat.reshape(7,60)
                          output = F.sigmoid(self.fc_net(s_reshape[i]))
                          self.aset.add(s_reshape[i].detach().numpy()[0])
                          #if(len(self.aset)>1):
                          #   print("s set len:",len(self.aset))
                          a = output.detach().numpy()
                          self.bset.add(a[0])
                          s_min = min(self.bset)
                          s_max = max(self.bset)
                          gap = (s_max - s_min) / 3
                          #if(len(self.bset)>1):
                          #   print("output set len:",len(self.bset))
                        else:
                          output = F.sigmoid(self.fc_net(s_flat))
                        if(output <= s_min + gap):
                           #print("output0:",output)
                           output = 0
                        elif(output <= s_min + 2 * gap):
                           #print("output1:",output)
                           output = 1
                        else:
                           #print("output2:",output)
                           output = 2
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
