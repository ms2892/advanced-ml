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
        scale = softplus_inverse(gain / torch.sqrt(torch.tensor(in_features)))

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


    def forward(self, x:torch.Tensor, n_samples=1):
        '''
            Calculates logits for the input x using "n_samples" samples of the weights.

            Args:
                x: tensor of shape (batch_size, in_features) if n_samples = 1
                   tensor of shape (batch_size, 1, 1, in_features) if n_samples > 1
            Output:
                tuple (tensor, scalar tensor):
                    - the first element is tensor of logits of the model for the input x.
                      If n_samples = 1, then the shape is (batch_size, out_features).
                      If n_samples > 1, then the shape is (batch_size, n_samples, out_features).
                    - the second element is the KL divergence of the model weights sampled
                      in the forward pass. It is a 0-dim tensor containing only one scalar.
        '''
        if n_samples == 1:
            return self.forward_single_sample(x)
        else:
            return self.forward_multiple_samples(x, n_samples=n_samples)


    def forward_single_sample(self, x:torch.Tensor):
        '''
            Args:
                x: input tensor of size (batch_size, in_features)
            Output:
                tuple (tensor, scalar tensor):
                    - the first element is tensor of logits of the model for the input x. The shape is (batch_size, out_features).
                    - the second element is the KL divergence of the model weights sampled
                      in the forward pass. It is a 0-dim tensor containing only one scalar.
        '''
        
        kl_divergence = 0

        # Calculate W
        weight_distribution = D.Normal(
            loc=self.mu_weights, scale=F.softplus(self.rho_weights)
        )
        W = weight_distribution.sample()

        # Calculate weight contribution to KL divergence
        kl_divergence += weight_distribution.log_prob(W).sum()
        kl_divergence -= self.prior_distribution.log_prob(W).sum()

        # Multiply input by W
        out = torch.mm(x, W.T)

        # Handle bias
        if self.bias:
            bias_distribution = D.Normal(
                loc=self.mu_bias, scale=F.softplus(self.rho_bias)
            )
            b = bias_distribution.sample()

            # Add the bias
            out += b

            # Calculate bias contribution to KL divergence
            kl_divergence += bias_distribution.log_prob(b).sum()
            kl_divergence -= self.prior_distribution.log_prob(b).sum()

        return out, kl_divergence
    

    def forward_multiple_samples(self, x:torch.Tensor, n_samples):
        '''
            Args:
                x: input tensor of size (batch_size, 1, 1, in_features)
            Output:
                tuple (tensor, scalar tensor):
                    - the first element is tensor of logits of the model for the input x. The shape is (batch_size, n_samples, out_features).
                    - the second element is the KL divergence of the model weights sampled
                      in the forward pass. It is a 0-dim tensor containing only one scalar.
        '''

        kl_divergence = 0

        # Calculate W
        weight_distribution = D.Normal(
            loc=self.mu_weights, scale=F.softplus(self.rho_weights)
        )
        W = weight_distribution.sample((n_samples, ))

        # Calculate weight contribution to KL divergence
        kl_divergence += weight_distribution.log_prob(W).sum()
        kl_divergence -= self.prior_distribution.log_prob(W).sum()

        # Multiply input by W
        out = torch.matmul(x, W.transpose(2, 1))

        # Handle bias
        if self.bias:
            bias_distribution = D.Normal(
                loc=self.mu_bias, scale=F.softplus(self.rho_bias)
            )
            b = bias_distribution.sample((n_samples, 1, ))

            # Add the bias
            out += b

            # Calculate bias contribution to KL divergence
            kl_divergence += bias_distribution.log_prob(b).sum()
            kl_divergence -= self.prior_distribution.log_prob(b).sum()

        return out, kl_divergence / n_samples


def main():
    p = 1/4
    mixture_distribution = D.Categorical(probs=torch.tensor([p, 1 - p]))
    component_distribution = D.Normal(loc=torch.zeros(2), scale=torch.tensor([0.1, 1]))
    prior_distribution = D.MixtureSameFamily(
        mixture_distribution=mixture_distribution, component_distribution=component_distribution
    )
    layer = VariationalLinear(
        in_features=784, out_features=1200, bias=True, prior_distribution=prior_distribution, nonlinearity="linear"
    )
    
    # for p in layer.parameters():
    #     print(p.shape)

    x = torch.randn((128, 784))
    with torch.no_grad():
        out, kl_divergence = layer(x)
    
    print(out.shape)
    print(out.mean(), out.std()) # Should be around 0 and 1

    print(kl_divergence) # Initial KL divergence
    print(2 * kl_divergence) # Twice initial KL divergence


    with torch.no_grad():
        total_kl_divergence = 0

        for _ in range(2):
            out, kl_divergence = layer(x)
            total_kl_divergence += kl_divergence

    print(total_kl_divergence) # Should be twice the initial


    with torch.no_grad():
        out, kl_divergence = layer(x)
    print(kl_divergence) # Should be approx. equal to initial again


    with torch.no_grad():
        layer.mu_weights += 0.1
        layer.rho_weights *= 0.5

        layer.mu_bias -= 0.1
        layer.rho_bias *= 0.1

    with torch.no_grad():
        out, kl_divergence = layer(x)
    print(kl_divergence) # Should be different from initial this time


    # Test with multiple samples
    print("\nMultiple samples")
    layer = VariationalLinear(
        in_features=784, out_features=1200, bias=True, prior_distribution=prior_distribution, nonlinearity="linear"
    )

    x = torch.randn((128, 1, 1, 784))
    with torch.no_grad():
        out, kl_divergence = layer(x, n_samples=5)
    
    print(out.shape)
    print(out.mean(), out.std()) # Should be around 0 and 1

    print(kl_divergence) # Should be around the same as initial

    # 2 samples
    with torch.no_grad():
        out, kl_divergence = layer(x, n_samples=2)
    
    print(kl_divergence) # Should be around the same as initial
    

if __name__ == "__main__":
    main()
