import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
import torch.distributions as D
from models.model_utils import BBB_Hyper, gaussian,mixture_prior,log_gaussian, log_gaussian_rho

class VariationalLayer(nn.Module):
    def __init__(self, n_input, n_output, hyper):
        super(VariationalLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.hyper = hyper
        self.s1 = hyper.s1
        self.s2 = hyper.s2
        self.pi = hyper.pi

        # We initialise weigth_mu and bias_mu as for usual Linear layers in PyTorch
        self.weight_mu = nn.Parameter(torch.Tensor(n_output, n_input))
        self.bias_mu = nn.Parameter(torch.Tensor(n_output))

        torch.nn.init.kaiming_uniform_(self.weight_mu, nonlinearity='relu')
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias_mu, -bound, bound)

        self.bias_rho = nn.Parameter(torch.Tensor(n_output).normal_(hyper.rho_init, .05))
        self.weight_rho = nn.Parameter(torch.Tensor(n_output, n_input).normal_(hyper.rho_init, .05))

        self.lpw = 0.
        self.lqw = 0.

    def forward(self, data, infer=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if infer:
            output = F.linear(data, self.weight_mu, self.bias_mu)
            return output

        epsilon_W = Variable(torch.Tensor(self.n_output, self.n_input).normal_(0, 1).to(device))
        epsilon_b = Variable(torch.Tensor(self.n_output).normal_(0, 1).to(device))
        W = self.weight_mu + torch.log(1+torch.exp(self.weight_rho)) * epsilon_W
        b = self.bias_mu + torch.log(1+torch.exp(self.bias_rho)) * epsilon_b

        output = F.linear(data, W, b)

        self.lqw = log_gaussian_rho(W, self.weight_mu, self.weight_rho).sum() + \
                   log_gaussian_rho(b, self.bias_mu, self.bias_rho).sum()

        if self.hyper.mixture:
            # print("the mixture is true")
            self.lpw = mixture_prior(W, self.pi, self.s2, self.s1).sum() + \
                       mixture_prior(b, self.pi, self.s2, self.s1).sum()
        else:
            self.lpw = log_gaussian(W, 0, self.s1).sum() + log_gaussian(b, 0, self.s1).sum()

        return output
    
if __name__=='__main__':
    test_class = BBB_Hyper()