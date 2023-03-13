import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D


class GATLayer(nn.Module):
    def __init__(self,in_features,out_features,n_heads=1,is_concat=True,dropout=0.6,leaky_relu=0.2):
        super(GATLayer,self).__init__()
        self.in_features = in_features
        self.n_heads = 1
        self.concat = is_concat
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
        self.W = nn.Parameter(torch.zeros(size=(in_features,self.n_hidden)))
        nn.init.xavier_uniform_(self.W.data,gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2*self.n_hidden,1)))
        nn.init.xavier_uniform_(self.a.data,gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(leaky_relu)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = dropout
        
    def forward(self,inp,edge_index):
        
        # print(edge_index)
        
        # print(torch.max(edge_index),torch.min(edge_index))
        # t=input()
        n_nodes = inp.shape[0]
        
        adj = torch.zeros(n_nodes,n_nodes)
        
        # print(adj.shape)
        
        for i in range(n_nodes):
            # print(edge_index[0,i].item(),edge_index[1,i].item())
            adj[edge_index[0,i].item(),edge_index[1,i].item()]=1
        h = torch.mm(inp,self.W)
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
        
def softplus_inverse(x):
    '''
        Computes the inverse of softplus f(x) = log(exp(x) - 1) in a numerically stable way.
    '''
    return x + torch.log(-torch.expm1(-x))
        

class VGATLayer(nn.Module):
    def __init__(self,in_features,out_features,n_heads,prior_distribution,nonlinearity='relu',bias=True,is_concat=True,dropout=True,leaky_relu=0.2):
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