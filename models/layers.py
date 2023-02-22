import torch
from torch import nn
import torch.nn.functional as F


def softplus_inverse(x):
    '''
        Computes the inverse of softplus f(x) = log(exp(x) - 1) in a numerically stable way.
    '''
    return x + torch.log(-torch.expm1(-x))


class VariationalLinear(nn.Module):
    def __init__(
        self,
        in_features:int, out_features:int, prior, bias=True,
        nonlinearity="relu", param=None,
    ):
        '''
            Args:
                prior:
                    the prior to be used.
                nonlinearity:
                    the nonlinearity that will follow the linear layer. This will be used to 
                    calculate the gain required to properly initialize the weights.
                    For more information see
                    https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain.
                    Default value is "relu".
                param:
                    any parameters needed to be passed for the function that calculates the gain
                    (see also the "nonlinearity" parameter).
        '''


        super().__init__()

        self.bias = bias

        # Calculate gain
        gain = torch.nn.init.calculate_gain(nonlinearity=nonlinearity, param=param)
        # Transform the gain according to the parameterization for sigma in the paper
        scale = torch.log(torch.exp(gain / torch.sqrt(torch.tensor(in_features))) - 1)
        print(scale)

        scale = softplus_inverse(gain / torch.sqrt(torch.tensor(in_features)))
        print(scale)

        # Note that initialization is so that initially the sampled weights follow
        # N(0, gain ** 2 / in_features) distribution. This helps retain output distribution to
        # be closer to standard normal
        self.mu_weights = nn.Parameter(torch.empty(out_features, in_features).fill_(0))
        self.rho_weights = nn.Parameter(torch.empty(out_features, in_features).fill_(scale))

        if bias:
            # Same initialization as above
            self.mu_bias = nn.Parameter(torch.empty(out_features).fill_(0))
            self.rho_bias = nn.Parameter(torch.empty(out_features).fill_(scale))
        
        self.prior = prior
    

    def forward(self, x:torch.Tensor):
        sigma = F.softplus(self.rho_weights)
        eps = torch.randn(self.mu_weights.shape)
        self.W = self.mu_weights + sigma * eps

        out = torch.mm(x, self.W.T)

        if self.bias:
            sigma = F.softplus(self.rho_bias)
            eps = torch.randn(self.mu_bias.shape)
            self.b = self.mu_bias + sigma * eps

            out += self.b

        return out
    

    def get_kl_divergence(self):
        torch.distribution


def main():
    prior = None
    layer = VariationalLinear(in_features=10000, out_features=1000, bias=True, prior=prior)
    
    for p in layer.parameters():
        print(p.shape)

    x = torch.randn((256, 10000))
    with torch.no_grad():
        out = layer(x)
    
    print(out.shape)
    

if __name__ == "__main__":
    main()
