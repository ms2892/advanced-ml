import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RegressionELBO(nn.Module):
    def __init__(self):
        super(RegressionELBO, self).__init__()

    def forward(self, output, label, kl_div, dataset_size=60000, batch_index=-1, weight_type='uniform'):
        batch_size = output.shape[0]
        M = dataset_size // batch_size
        if weight_type == 'uniform':
            kl_weight = 1 / M
        else:
            if batch_index == -1:
                raise Exception("Batch Index Not specified while getting Loss")
            kl_weight = 2**(M-batch_index)/(2**M-1)
        nll = self.get_neg_log_lik(output, label)
        elbo = kl_weight*kl_div/dataset_size + nll
        return elbo, nll

    def get_neg_log_lik(y_pred, y_true):
        sigma=1
        batched_nll = (y_pred - y_true.unsqueeze(-1))**2/(2*sigma**2) + torch.log(torch.tensor(sigma))
        return (batched_nll.mean(dim=0)).mean(dim=0)
        



class Regression(nn.Module):
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

    def __init__(self, input_dim, output_dim, hl_type,hl_units):
        '''
            Constructor:
                This method describes the constructor for the network

            Args:
                input_dim   :   describes number of input nodes to have
                output_dim  :   describes number of output nodes to have
                hidden_layer:   describes number of hidden layer nodes to have
        '''
        super(Regression,self).__init__()
        self.inp = hl_type(input_dim, hl_units)
        self.linear1 = hl_type(hl_units, hl_units)
        self.out = hl_type(hl_units, output_dim)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.double()

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


class Regression_Dropout(nn.Module):
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
        super(Regression_Dropout,self).__init__()
        self.inp = hl_type(input_dim, hl_units)
        self.linear1 = hl_type(hl_units, hl_units)
        self.out = hl_type(hl_units, output_dim)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.double()

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
        x = F.dropout(x, p=0.2)
        
        intermediate = self.inp(x)
        # 50% Dropout at 1st linear layer
        intermediate = F.dropout(intermediate, p=0.5)
        intermediate = self.relu(intermediate)
        
        # 50 % dropout at 2nd linear layer
        intermediate = self.linear1(intermediate)
        intermediate = F.dropout(intermediate, p=0.5)
        intermediate = self.relu(intermediate)
        
        # Output Nodes
        output = self.out(intermediate)
        return output