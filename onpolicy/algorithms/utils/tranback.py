import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class BackNet(nn.Module):
    def __init__(self, device):
        super(BackNet, self).__init__()
        self.device = device

    def transback(self, inputs):
        for i in range (0,len(inputs)):
            for j in range (0,len(inputs[i])):
                if inputs[i][j] >= 6:
                    inputs[i][j] = 6
        return inputs


if __name__ == '__main__':
    pass
