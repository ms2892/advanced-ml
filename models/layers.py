import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D


def softplus_inverse(x):
    '''
        Computes the inverse of softplus f(x) = log(exp(x) - 1) in a numerically stable way.
    '''
    return x + torch.log(-torch.expm1(-x))


class VariationalLinear(nn.Module):
    def __init__(
        self,
        in_features:int, out_features:int, prior_distribution, bias=True,
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
        
        self.prior_distribution = prior_distribution

        self.kl_divergence = 0
    

    def forward(self, x:torch.Tensor):
        # Calculate W
        self.weight_distribution = D.Normal(
            loc=self.mu_weights, scale=F.softplus(self.rho_weights)
        )
        self.W = self.weight_distribution.sample()

        # Multiply input by W
        out = torch.mm(x, self.W.T)

        # Handle bias
        if self.bias:
            self.bias_distribution = D.Normal(
                loc=self.mu_bias, scale=F.softplus(self.rho_bias)
            )
            self.b = self.bias_distribution.sample()

            out += self.b
        
        # Update KL divergence
        self.update_kl_divergence()

        return out
    

    def update_kl_divergence(self):
        self.kl_divergence += self.weight_distribution.log_prob(self.W).sum()
        self.kl_divergence -= self.prior_distribution.log_prob(self.W).sum()

        if self.bias:
            self.kl_divergence += self.weight_distribution.log_prob(self.W).sum()
            self.kl_divergence -= self.prior_distribution.log_prob(self.W).sum()
    

    def reset_kl_divergence(self):
        self.kl_divergence = 0


def main():
    p = 1/4
    mixture_distribution = D.Categorical(probs=torch.tensor([p, 1 - p]))
    component_distribution = D.Normal(loc=torch.zeros(2), scale=torch.tensor([0.1, 1]))
    prior_distribution = D.MixtureSameFamily(
        mixture_distribution=mixture_distribution, component_distribution=component_distribution
    )
    layer = VariationalLinear(
        in_features=10000, out_features=1000, bias=True, prior_distribution=prior_distribution, nonlinearity="linear"
    )
    
    for p in layer.parameters():
        print(p.shape)

    x = torch.randn((256, 10000))
    with torch.no_grad():
        out = layer(x)
    
    print(out.shape)
    print(out.mean(), out.std()) # Should be around 0 and 1

    print(layer.kl_divergence) # Initial KL divergence
    print(2 * layer.kl_divergence) # Twice initial KL divergence


    with torch.no_grad():
        out = layer(x)
    print(layer.kl_divergence) # Should be twice the initial


    layer.reset_kl_divergence()
    with torch.no_grad():
        out = layer(x)
    print(layer.kl_divergence) # Should be approx. equal to initial again
    

if __name__ == "__main__":
    main()
