import torch
from torch import nn


class VariationalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.bias = bias

        self.mu_weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.rho_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        # Initialize

        if bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_features))
            self.rho_bias = nn.Parameter(torch.Tensor(out_features))



def main():
    pass


if __name__ == "__main__":
    main()
