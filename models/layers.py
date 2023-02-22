import torch
from torch import nn


class VariationalLinear(nn.Module):
    def __init__(self, in_features:int, out_features:int, bias=True, nonlinearity="relu", param=None):
        '''
            nonlinearity:
                the nonlinearity that will follow the linear layer. This will be used to 
                calculate the gain required to properly initialize the weights.
                For more information see
                https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain.
                Default value is "relu".
            param:
                any parameters needed to be passed for the function that calculates the gain
                (see also the "nonlinearity" parameter)
        '''


        super().__init__()

        self.bias = bias

        # Calculate gain
        gain = torch.nn.init.calculate_gain(nonlinearity=nonlinearity, param=param)
        print(gain)
        # Transform the scale of gain
        scale = torch.log(torch.exp(gain / torch.sqrt(torch.tensor(in_features))) - 1)
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
    

    def forward(self, x:torch.Tensor):
        sigma = (self.rho_weights.exp() + 1).log()
        eps = torch.randn(self.mu_weights.shape)
        W = self.mu_weights + sigma * eps

        out = torch.mm(x, W.T)

        if self.bias:
            sigma = (self.rho_bias.exp() + 1).log()
            eps = torch.randn(self.mu_bias.shape)
            b = self.mu_bias + sigma * eps

            out += b

        return out


def main():
    import torch.nn.functional as F

    # layer = VariationalLinear(in_features=10000, out_features=1000, bias=True)
    layer = VariationalLinear(in_features=10000, out_features=1000, bias=False, nonlinearity="relu")

    x = torch.randn((256, 10000))
    with torch.no_grad():
        out = layer(x)
        out = F.relu(out)

    print(out.mean(), out.std())

    
    layer = VariationalLinear(in_features=10000, out_features=1000, bias=False, nonlinearity="tanh")

    x = torch.randn((256, 10000))
    with torch.no_grad():
        out = layer(x)
        out = torch.tanh(out)

    print(out.mean(), out.std())
    

if __name__ == "__main__":
    main()
