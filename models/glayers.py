import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D

from models.layers import VariationalLinear, softplus_inverse
# from layers import VariationalLinear, softplus_inverse


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1, is_concat=True, dropout=0.6, leaky_relu=0.2):
        super().__init__()

        self.in_features = in_features
        self.n_heads = n_heads
        
        self.concat = is_concat

        assert out_features % n_heads == 0
        self.hidden_features = out_features // n_heads
        
        self.W = nn.Linear(in_features, self.hidden_features * n_heads, bias=False)
        nn.init.xavier_uniform_(self.W.weight) # there was no gain in the original implementation
        
        # Avoid concat by applying a_left to the 1st vector and a_right to the 2nd
        self.a_left = nn.Parameter(torch.Tensor(1, n_heads, self.hidden_features))
        nn.init.xavier_uniform_(self.a_left) # there was no gain in the original implementation
        self.a_right = nn.Parameter(torch.Tensor(1, n_heads, self.hidden_features))
        nn.init.xavier_uniform_(self.a_right) # there was no gain in the original implementation
        
        self.leaky_relu = nn.LeakyReLU(leaky_relu)
        self.dropout = dropout


    def forward(self, inp, A):
        '''
            A: adjacency matrix
        '''

        # Paper uses dropout on input
        inp = F.dropout(inp, p=self.dropout, training=self.training)
        
        # (n_heads, N, hidden_features)
        h = self.W(inp).view(-1, self.n_heads, self.hidden_features)

        logits_source = torch.sum(h * self.a_left, dim=-1, keepdim=True).transpose(0, 1) # (n_heads, N, 1)
        logits_target = torch.sum(h * self.a_left, dim=-1, keepdim=True).permute(1, 2, 0) # (n_heads, 1, N)

        attention_coeffs = self.leaky_relu(logits_source + logits_target)
        attention_coeffs = attention_coeffs + A
        attention_coeffs = F.softmax(attention_coeffs, dim=-1)

        # Apply dropout
        attention_coeffs = F.dropout(attention_coeffs, p=self.dropout, training=self.training)

        out_node_features = torch.bmm(attention_coeffs, h.transpose(0, 1))
        out_node_features = out_node_features.transpose(0, 1)
        
        if self.concat:
            # Concat over head dimension
            out_node_features = out_node_features.reshape(-1, self.hidden_features * self.n_heads)
        else:
            # Mean over head dimension
            out_node_features = out_node_features.mean(dim=1)
        
        return out_node_features


class VGATLayer(nn.Module):
    def __init__(
            self, in_features, out_features, prior_distribution,
            n_heads=1, is_concat=True, dropout=0.6, leaky_relu=0.2,
        ):
        super().__init__()

        self.in_features = in_features
        self.n_heads = n_heads
        
        self.concat = is_concat

        assert out_features % n_heads == 0
        self.hidden_features = out_features // n_heads
        
        self.W = VariationalLinear(
            in_features, self.hidden_features * n_heads,
            prior_distribution=prior_distribution, bias=False,
        )
        # Below is the a from Uniform(-a, a) random variable
        a = (6 / (in_features + self.hidden_features * n_heads))**0.5
        scale = (a**2 / 3)**0.5 # this is the best scale to approximate uniform with normal
        scale = softplus_inverse(torch.tensor(scale))
        with torch.no_grad():
            self.W.rho_weights.normal_(scale, 0.1) # there was no gain in the original implementation
        
        # Avoid concat by applying a_left to the 1st vector and a_right to the 2nd
        self.a_left = VariationalLinear(
            self.hidden_features, 1,
            prior_distribution=prior_distribution, bias=False
        )
        self.a_right = VariationalLinear(
            self.hidden_features, 1,
            prior_distribution=prior_distribution, bias=False
        )
        a = (6 / (in_features + 1))**0.5
        scale = (a**2 / 3)**0.5 # this is the best scale to approximate uniform with normal
        scale = softplus_inverse(torch.tensor(scale))
        with torch.no_grad():
            self.a_left.rho_weights.normal_(scale, 0.1) # there was no gain in the original implementation
            self.a_right.rho_weights.normal_(scale, 0.1) # there was no gain in the original implementation
        
        self.leaky_relu = nn.LeakyReLU(leaky_relu)
        # self.dropout = dropout # No dropout for now


    def forward(self, inp, A):
        '''
            A: adjacency matrix
        '''
        kl_divergence = 0

        # Paper uses dropout on input
        # inp = F.dropout(inp, p=self.dropout, training=self.training)
        
        # (n_heads, N, hidden_features)
        h, tmp_kl_divergence = self.W(inp)
        kl_divergence += tmp_kl_divergence
        h = h.view(self.n_heads, -1, self.hidden_features)


        logits_source, tmp_kl_divergence = self.a_left(h) # (n_heads, N, 1)
        kl_divergence += tmp_kl_divergence

        logits_target, tmp_kl_divergence = self.a_right(h) # (n_heads, 1, N)
        logits_target = logits_target.transpose(1, 2)
        kl_divergence += tmp_kl_divergence

        attention_coeffs = self.leaky_relu(logits_source + logits_target)
        attention_coeffs = attention_coeffs + A
        attention_coeffs = F.softmax(attention_coeffs, dim=-1)

        # Apply dropout
        # attention_coeffs = F.dropout(attention_coeffs, p=self.dropout, training=self.training)

        out_node_features = torch.bmm(attention_coeffs, h)
        out_node_features = out_node_features.transpose(0, 1)
        
        if self.concat:
            # Concat over head dimension
            out_node_features = out_node_features.reshape(-1, self.hidden_features * self.n_heads)
        else:
            # Mean over head dimension
            out_node_features = out_node_features.mean(dim=1)
        
        return out_node_features, kl_divergence
        

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GATLayer(
        in_features=1433, out_features=64,
        n_heads=8,
        is_concat=True,
    ).to(device)

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
                
    with torch.no_grad():
        output = model(x, A)
    print(output.shape)



    # Test variational
    import torch.distributions as D
    prior_distribution = D.Normal(
        loc=torch.zeros(1).to(device),
        scale=torch.tensor(1).to(device),
    )
    model = VGATLayer(
        in_features=1433, out_features=64, n_heads=8,
        prior_distribution=prior_distribution,
    )
    with torch.no_grad():
        output, kl_divergence = model(x, A)
    print(output.shape)
    print(kl_divergence)

