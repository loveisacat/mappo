import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .transformer import Transformer

#This is the Dynamic Attention Representation Network(DarNet)
class DarNet(nn.Module):
    def __init__(self, agent_num, output_shape, device):
        super(DarNet, self).__init__()
        self.enemy_shape = 5
        self.ally_shape = 5
        self.move_direction = 4
        self.unit_shape = 1
        self.n_actions = 6
        self.n_agent = agent_num
        self.n_enemy = agent_num
        self.n_ally = agent_num
        #heads must be 1,2,3,4 ... which can be divided by (5 + 5 + 6)
        self.heads = 1
        self.device = device

        #Init the attention encoder
        self.dyn_attention = Transformer(self.enemy_shape + self.ally_shape + self.n_actions, self.enemy_shape + self.ally_shape + self.n_actions, self.heads)
        #Init the simple NN encoder
        #self.fc_net = nn.Linear(self.enemy_shape + self.ally_shape + self.n_actions, self.enemy_shape + self.ally_shape + self.n_actions)
         

    def forward(self, inputs):
        #The original inputs structure is : move_direction, agent_num * enemy_shape, (agent_num - 1) * ally_shape, own_shape,  agent_num
        enemy_inputs = inputs[:,:,self.move_direction:self.move_direction + self.enemy_shape * self.n_enemy]
        enemy_inputs = enemy_inputs.reshape(enemy_inputs.shape[0],enemy_inputs.shape[1],self.n_enemy,self.enemy_shape)
        ally_inputs = inputs[:,:,self.move_direction + self.enemy_shape * self.n_enemy:self.move_direction + self.enemy_shape * self.n_enemy + (self.ally_shape+self.n_actions+self.n_agent) * self.n_ally]
        ally_inputs = ally_inputs.reshape(ally_inputs.shape[0], ally_inputs.shape[1], self.n_ally, self.ally_shape+self.n_actions+self.n_agent)
        
        dyn_inputs = inputs[:,:,self.move_direction:self.move_direction + self.enemy_shape * self.n_enemy + (self.ally_shape + self.n_actions + self.n_agent) * self.n_ally]
        dyn_inputs = dyn_inputs.reshape(dyn_inputs.shape[0], dyn_inputs.shape[1], self.n_agent, self.ally_shape + self.enemy_shape + self.n_actions + self.n_agent)
        dyn_inputs = dyn_inputs[:,:,:,:(self.enemy_shape+self.ally_shape+self.n_actions)]
        #enemy_inputs = [F.relu(self.enemy_attention(enemy_inputs[i])) for i in range(self.n_enemy)]
        #ally_inputs = [F.relu(self.ally_attention(ally_inputs[i])) for i in range(self.n_ally)]

        dyn_inputs = torch.from_numpy(dyn_inputs)
        dyn_outputs = torch.empty([dyn_inputs.shape[0],dyn_inputs.shape[1],self.ally_shape + self.enemy_shape + self.n_actions])
        dyn_inputs = dyn_inputs.to(self.device)

        for j in range(dyn_inputs.shape[0]):
          k_list = [F.relu(self.dyn_attention(dyn_inputs[j][i])) for i in range(self.n_agent)]
          #k_list = [F.relu(self.fc_net(dyn_inputs[j][i])) for i in range(self.n_agent)]
          #For sum aggregate
          x_sum = torch.zeros_like(k_list[0])
          for i in range(self.n_agent):
              x_sum += k_list[i]
          dyn_outputs[j] = x_sum
        dyn_outputs = dyn_outputs.detach().numpy()
        dyn_outputs = np.concatenate((inputs[:,:,:self.move_direction],dyn_outputs),axis=2)

        return dyn_outputs

if __name__ == '__main__':
    pass
