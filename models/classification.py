import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import VariationalLinear


class CrossEntropyELBO(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels, kl_divergence, kl_weight):
        nll = self._get_neg_log_lik(y_pred=outputs, y_true=labels)

        elbo = kl_weight * kl_divergence + nll
        
        return elbo, nll


    def _get_neg_log_lik(self, y_pred, y_true):
        n_samples = y_pred.shape[1]

        out = F.cross_entropy(
            y_pred.transpose(2, 1),
            y_true.unsqueeze(-1).repeat(repeats=(1, n_samples)),
            reduction="none"
        )

        return out.sum(dim=0).mean(dim=0)


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
        super(Classification, self).__init__()
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

    def __init__(self, input_dim, output_dim, hl_type, hl_units):
        '''
            Constructor:
                This method describes the constructor for the network

            Args:
                input_dim   :   describes number of input nodes to have
                output_dim  :   describes number of output nodes to have
                hidden_layer:   describes number of hidden layer nodes to have
        '''
        super(Classification_Dropout, self).__init__()
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


class VariationalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, prior_distribution):
        super().__init__()

        self.prior_distribution = prior_distribution

        self.layers = nn.ModuleList([
            VariationalLinear(
                in_features=input_dim, out_features=hidden_dim,
                prior_distribution=prior_distribution
            ),
            VariationalLinear(
                in_features=hidden_dim, out_features=hidden_dim,
                prior_distribution=prior_distribution
            ),
            VariationalLinear(
                in_features=hidden_dim, out_features=output_dim,
                prior_distribution=prior_distribution
            ),
        ])
    
    def _single_forward(self, x):
        total_kl_divergence = 0
        for i, layer in enumerate(self.layers):
            x, kl_divergence = layer(x)
            
            if i < len(self.layers) - 1:
                x = F.relu(x)

            total_kl_divergence += kl_divergence

        x = x.unsqueeze(dim=1)

        return x, total_kl_divergence
    

    def forward(self, x, n_samples=1):
        logits = []
        kl_divergence = 0
        
        for _ in range(n_samples):
            curr_logits, curr_kl_divergence = self._single_forward(x)
            logits.append(curr_logits)
            kl_divergence += curr_kl_divergence

        logits = torch.cat(logits, axis=1)
        kl_divergence /= n_samples
    
        return logits, kl_divergence


def main():
    import torch
    import torch.distributions as D

    sigma_1 = torch.exp(-torch.tensor(1))
    sigma_2 = torch.exp(-torch.tensor(7))

    p = 1/2
    mixture_distribution = D.Categorical(probs=torch.tensor([p, 1 - p]))
    component_distribution = D.Normal(
        loc=torch.zeros(2),
        scale=torch.tensor([sigma_1, sigma_2]),
    )
    prior_distribution = D.MixtureSameFamily(
        mixture_distribution=mixture_distribution, component_distribution=component_distribution
    )

    model = VariationalMLP(input_dim=784, hidden_dim=800, output_dim=10, prior_distribution=prior_distribution)
    
    for p in model.parameters():
        print(p.shape)

    x = torch.randn((128, 784))
    with torch.no_grad():
        out, kl_divergence = model(x)

    print(out.shape)
    print(kl_divergence)

    # Test with multiple samples
    print("\nMultiple samples")
    with torch.no_grad():
        out, kl_divergence = model(x, n_samples=5)

    print(out.shape)
    print(kl_divergence)


if __name__ == "__main__":
    main()
