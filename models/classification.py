import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPModel(nn.Module):
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

    def __init__(self, input_dim, output_dim, hidden_layer):
        '''
            Constructor:
                This method describes the constructor for the network

            Args:
                input_dim   :   describes number of input nodes to have
                output_dim  :   describes number of output nodes to have
                hidden_layer:   describes number of hidden layer nodes to have
        '''
        super(MLPModel,self).__init__()
        self.inp = nn.Linear(input_dim, hidden_layer)
        self.linear1 = nn.Linear(hidden_layer, hidden_layer)
        self.out = nn.Linear(hidden_layer, output_dim)
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


class MLPModel_Dropout(nn.Module):
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

    def __init__(self, input_dim, output_dim, hidden_layer):
        '''
            Constructor:
                This method describes the constructor for the network

            Args:
                input_dim   :   describes number of input nodes to have
                output_dim  :   describes number of output nodes to have
                hidden_layer:   describes number of hidden layer nodes to have
        '''
        super(MLPModel_Dropout,self).__init__()
        self.inp = nn.Linear(input_dim, hidden_layer)
        self.linear1 = nn.Linear(hidden_layer, hidden_layer)
        self.out = nn.Linear(hidden_layer, output_dim)
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
        intermediate = self.inp(x)
        
        # 20% Dropout at input layer
        intermediate = F.dropout(intermediate, p=0.2)
        intermediate = self.relu(intermediate)
        intermediate = self.linear1(intermediate)
        
        # 50% Dropout at 1st linear layer
        intermediate = F.dropout(intermediate, p=0.5)
        intermediate = self.relu(intermediate)
        
        # 50 % dropout at 2nd linear layer
        output = self.out(intermediate)
        output = F.dropout(output, p=0.5)
        return output
