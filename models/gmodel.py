from models.glayers import GATLayer, VGATLayer

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATELBO(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, y_true, kl_divergence):
        nll = self._get_neg_log_lik(logits=logits, y_true=y_true)

        elbo = kl_divergence + nll
        
        return elbo, nll


    def _get_neg_log_lik(self, logits, y_true):
        n_samples = logits.shape[1]

        out = F.cross_entropy(
            logits.transpose(2, 1),
            y_true.unsqueeze(-1).repeat(repeats=(1, n_samples)),
            reduction="none"
        )

        return out.sum(dim=0).mean(dim=0)


class GAT(nn.Module):
    
    def __init__(
        self, in_features, hidden_features, num_classes,
        n_heads=1, dropout=0.6, leaky_relu=0.2
    ):
        super(GAT, self).__init__()

        self.gat1 = GATLayer(
            in_features, hidden_features, n_heads,
            dropout=dropout, leaky_relu=leaky_relu
        )
        self.gat2 = GATLayer(
            hidden_features, num_classes, 1,
            dropout=dropout, is_concat=False, leaky_relu=leaky_relu
        )


    def forward(self, x, A):
        x = self.gat1(x, A)
        x = F.elu(x)
        x = self.gat2(x, A)
        
        return x
    

class VGAT(nn.Module):
    
    def __init__(
        self, in_features, hidden_features, num_classes,
        prior_distribution, n_heads=1, dropout=0.6, leaky_relu=0.2
    ):
        super(VGAT, self).__init__()

        self.prior_distribution = prior_distribution
        
        self.vgat1 = VGATLayer(
            in_features, hidden_features, n_heads=n_heads,
            prior_distribution=prior_distribution,
            dropout=dropout, leaky_relu=leaky_relu
        )
        self.vgat2 = VGATLayer(
            hidden_features, num_classes, n_heads=1,
            prior_distribution=prior_distribution,
            dropout=dropout, is_concat=False, leaky_relu=leaky_relu
        )
    

    def _single_forward(self, x, A):
        total_kl_divergence = 0

        x, tmp_kl_divergence = self.vgat1(x, A)
        total_kl_divergence += tmp_kl_divergence

        x = F.elu(x)

        x, tmp_kl_divergence = self.vgat2(x, A)
        total_kl_divergence += tmp_kl_divergence

        x = x.unsqueeze(dim=1)

        return x, total_kl_divergence
    

    def forward(self, x, A, n_samples=1):
        logits = []
        kl_divergence = 0
        
        for _ in range(n_samples):
            curr_logits, curr_kl_divergence = self._single_forward(x, A)
            logits.append(curr_logits)
            kl_divergence += curr_kl_divergence

        logits = torch.cat(logits, axis=1)
        kl_divergence /= n_samples
    
        return logits, kl_divergence


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Features
    x = torch.rand(2708, 1433).to(device)

    # Adjacency matrix
    edge_index = torch.randint(2708, size=(2, 10566)).to(device)
    A = 1.0 - torch.sparse.LongTensor(
        edge_index, # where to put
        torch.ones(edge_index.shape[1]).to(device),
        torch.Size((x.shape[0], x.shape[0])),
    ).to_dense()
    A[A.bool()] = float("-Inf")
    A = A.unsqueeze(0)


    # Test variational
    import torch.distributions as D
    prior_distribution = D.Normal(
        loc=torch.zeros(1).to(device),
        scale=torch.tensor(1).to(device),
    )
    model = VGAT(
        in_features=1433, hidden_features=64, num_classes=7,
        n_heads=8, prior_distribution=prior_distribution,
    )
    with torch.no_grad():
        output, kl_divergence = model(x, A, n_samples=5)
    print(output.shape)
    print(kl_divergence)
