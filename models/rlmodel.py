import torch
import torch.nn as nn

import numpy as np

class RLModel:
    def __init__(self,inp_units,hl_units,output_units):
        super(RLModel,self).__init__()
        self.inp = nn.Linear(inp_units,hl_units)
        self.hidden1 = nn.Linear(hl_units,hl_units)
        self.output = nn.Linear(hl_units,output_units)
        self.relu = nn.ReLU()

    def forward(self,x):
        intermediate = self.inp(x)
        intermediate = self.relu(intermediate)
        
        intermediate = self.hidden1(intermediate)
        intermediate = self.relu(intermediate)
        
        output = self.output(intermediate)
        output = self.relu(output)
        
        return output

class Agent:
    def __init__():
        