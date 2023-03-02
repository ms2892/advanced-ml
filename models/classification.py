import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers import VariationalLayer


class Classification(nn.Module):
    '''
        Class:
            This class is meant to mimic the Artificial Neural Network used in the paper
            Weight Uncertainty in Neural Networks for MNIST Classification

        Attributes:
            inp     :   describes the linear transformation from inp size to a hidden layer of x units
            linear1 :   describes the linear transformation from hidden layer of x units to hidden layer of x units 
            out     :   describes the linear transformation from hidden layer of x units to num_classes units
            relu    :   Wrapper function to apply ReLU activation function

        Methods:
            forward :   defines the forward pass of the network
    '''

    def __init__(self, input_dim, output_dim, hl_type, hl_units):
        '''
            Constructor:
                This method describes the constructor for the network

            Args:
                input_dim   :   describes number of input nodes to have
                output_dim  :   describes number of output nodes to have
                hidden_layer:   describes number of hidden layer nodes to have
        '''
        super(Classification,self).__init__()
        self.inp = hl_type(input_dim, hl_units)
        self.linear1 = hl_type(hl_units, hl_units)
        self.out = hl_type(hl_units, output_dim)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        '''
            Method:
                This method describes the forward pass in a network

            Args:
                x   :   input tensor

            Output:
                (tensor)    :   Output of the model after performing the forward pass
        '''
        x = self.flatten(x)
        # print(x.shape)
        # t=input()
        intermediate = self.inp(x)
        intermediate = self.relu(intermediate)
        intermediate = self.linear1(intermediate)
        intermediate = self.relu(intermediate)
        output = self.out(intermediate)
        return output


class Classification_Dropout(nn.Module):
    '''
        Class:
            This class is meant to mimic the Artificial Neural Network with dropout used in the paper
            Weight Uncertainty in Neural Networks for MNIST Classification

        Attributes:
            inp     :   describes the linear transformation from inp size to a hidden layer of x units
            linear1 :   describes the linear transformation from hidden layer of x units to hidden layer of x units 
            out     :   describes the linear transformation from hidden layer of x units to num_classes units
            relu    :   Wrapper function to apply ReLU activation function

        Methods:
            forward :   defines the forward pass of the network
    '''

    def __init__(self, input_dim, output_dim, hl_type,hl_units):
        '''
            Constructor:
                This method describes the constructor for the network

            Args:
                input_dim   :   describes number of input nodes to have
                output_dim  :   describes number of output nodes to have
                hidden_layer:   describes number of hidden layer nodes to have
        '''
        super(Classification_Dropout,self).__init__()
        self.inp = hl_type(input_dim, hl_units)
        self.linear1 = hl_type(hl_units, hl_units)
        self.out = hl_type(hl_units, output_dim)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)


    def forward(self, x):
        '''
            Method:
                This method describes the forward pass in a network

            Args:
                x   :   input tensor

            Output:
                (tensor)    :   Output of the model after performing the forward pass
        '''
        x = self.flatten(x)

        # 20% Dropout at input layer
        x = self.drop1(x)
#         x = F.dropout(x, p=0.2)
#         print(x.shape)
        intermediate = self.inp(x)
        # 50% Dropout at 1st linear layer
        intermediate = self.drop2(intermediate)
        intermediate = self.relu(intermediate)
#         print(intermediate.shape)
        
        # 50 % dropout at 2nd linear layer
        intermediate = self.linear1(intermediate)
        intermediate = self.drop2(intermediate)
        intermediate = self.relu(intermediate)
        
        # Output Nodes
        output = self.out(intermediate)
        return output


class VariationalModel(nn.Module):
    def __init__(self, n_input, n_ouput, hyper):
        super(VariationalModel, self).__init__()

        self.n_input = n_input
        self.layers = nn.ModuleList([])
        self.layers.append(VariationalLayer(n_input, hyper.hidden_units, hyper))
        self.layers.append(VariationalLayer(hyper.hidden_units, hyper.hidden_units, hyper))
        self.layers.append(VariationalLayer(hyper.hidden_units, n_ouput, hyper))

    def forward(self, data, infer=False):
        output = F.relu(self.layers[0](data.view(-1, self.n_input), infer))
        output = F.relu(self.layers[1](output, infer))
        output = F.softmax(self.layers[2](output, infer), dim=1)
        return output

    def get_lpw_lqw(self):
        lpw = self.layers[0].lpw + self.layers[1].lpw + self.layers[2].lpw
        lqw = self.layers[0].lqw + self.layers[1].lqw + self.layers[2].lqw
        return lpw, lqw

    