import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D

from models.layers import VariationalLinear
# from layers import VariationalLinear


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
        self.a_left = nn.Linear(self.hidden_features, 1, bias=False)
        nn.init.xavier_uniform_(self.a_left.weight) # there was no gain in the original implementation
        self.a_right = nn.Linear(self.hidden_features, 1, bias=False)
        nn.init.xavier_uniform_(self.a_right.weight) # there was no gain in the original implementation
        
        self.leaky_relu = nn.LeakyReLU(leaky_relu)
        self.dropout = dropout


    def forward(self, inp, A):
        '''
            A: adjacency matrix
        '''

        # Paper uses dropout on input
        inp = F.dropout(inp, p=self.dropout, training=self.training)
        
        # (n_heads, N, hidden_features)
        h = self.W(inp).view(self.n_heads, -1, self.hidden_features)


        logits_source = self.a_left(h) # (n_heads, N, 1)
        logits_target = self.a_right(h).transpose(1, 2) # (n_heads, 1, N)
        attention_coeffs = self.leaky_relu(logits_source + logits_target)
        attention_coeffs = attention_coeffs + A
        attention_coeffs = F.softmax(attention_coeffs, dim=-1)

        # Apply dropout
        attention_coeffs = F.dropout(attention_coeffs, p=self.dropout, training=self.training)

        out_node_features = torch.bmm(attention_coeffs, h)
        out_node_features = out_node_features.transpose(0, 1)
        
        if self.concat:
            # Concat over head dimension
            out_node_features = out_node_features.reshape(-1, self.hidden_features * self.n_heads)
        else:
            # Mean over head dimension
            out_node_features = out_node_features.mean(dim=1)
        
        return out_node_features


def softplus_inverse(x):
    '''
        Computes the inverse of softplus f(x) = log(exp(x) - 1) in a numerically stable way.
    '''
    return x + torch.log(-torch.expm1(-x))
        

class VGATLayer(nn.Module):
    def __init__(
            self, in_features, out_features, n_heads,
            prior_distribution,nonlinearity='relu', bias=True,is_concat=True,dropout=True,leaky_relu=0.2):
        super(GATLayer,self).__init__()
        
        
        self.in_features = in_features
        self.n_heads = 1
        self.is_concat = is_concat
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
        self.leaky_relu = nn.LeakyReLU(leaky_relu)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = dropout
        
        self.bias = bias
        gain = nn.init.calculate_gain(nonlinearity=nonlinearity)
        scale = softplus_inverse(gain/torch.sqrt(torch.tensor(in_features)))
        
        self.mu_weights = nn.Parameter(torch.empty(self.n_hidden*n_heads,in_features).fill_(0))
        self.rho_weights = nn.Parameter(torch.empty(self.n_hidden*n_heads,in_features).fill_(scale))

        self.W = VariationalLinear(
            in_features=in_features, out_features=self.n_hidden * n_heads,
            prior_distribution=prior_distribution,
            bias=False, # the usual GAT does not use a bias
        )
        
        
        self.a = nn.Parameter(torch.empty(1,2*self.n_hidden).fill_(0))
        nn.init.xavier_uniform_(self.a.data,gain=1.414)
        
        self.prior_distribution = prior_distribution
        
        
    def forward(self,inp,edge_index):
        
        kl_divergence= 0
        
        weight_distribution = D.Normal(
            loc = self.mu_weights,scale=F.softplus(self.rho_weights)
        )
        
        W = weight_distribution.rsample()
        
        kl_divergence += weight_distribution.log_prob(W).sum()
        kl_divergence -= self.prior_distribution.log_prob(W).sum()
    
        # -----------------------------------------------------------#
        
        n_nodes = inp.shape[0]
        
        adj = torch.zeros(n_nodes,n_nodes)
        
        # print(adj.shape)
        
        for i in range(n_nodes):
            # print(edge_index[0,i].item(),edge_index[1,i].item())
            adj[edge_index[0,i].item(),edge_index[1,i].item()]=1
        h = torch.mm(inp,self.W)
        if self.bias:
            bias_distribution = D.Normal(
                loc=self.mu_bias,scale=F.softplus(self.rho_bias)
            )
            b = bias_distribution.rsample()
            
            h+=b
            
            kl_divergence += bias_distribution.log_prob(b).sum()
            kl_divergence -= self.prior_distribution.log_prob(b).sum()
        
        N = h.size()[0] # N is the number of nodes in the graph
        
        # Attention Mechanism
        a_input = torch.cat([h.repeat(1,N).view(N*N,-1),h.repeat(N,1)],dim=1).view(N,-1,2*self.n_hidden)
        
        
            
            
        e = self.leaky_relu(torch.matmul(a_input,self.a).squeeze(2))
        
        # Masked Attnetion
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj>0,e,zero_vec)
        
        attention = F.softmax(attention,dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)   # self.training = True or False depending on the mode of the model 
        h_prime = torch.matmul(attention,h)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        

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
