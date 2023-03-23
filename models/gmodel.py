from models.glayers import GATLayer,VGATLayer
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATELBO(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs, labels, kl_div, kl_weight):
        nll = F.nll_loss(outputs, labels)
        
        elbo = kl_weight * kl_div + nll
        
        return elbo,nll


class GAT(nn.Module):
    
    def __init__(self, in_features, hidden_features, num_classes, n_heads=1, dropout=0.6, leaky_relu=0.2):
        super(GAT, self).__init__()
        
        self.gat1 = GATLayer(
            in_features, hidden_features, n_heads,
            dropout=dropout, leaky_relu=leaky_relu
        )
        self.gat2 = GATLayer(
            hidden_features, num_classes, 1,
            dropout=dropout, is_concat=False, leaky_relu=leaky_relu
        )


    def forward(self, data):
        x, edge_index = data.x,data.edge_index
        
        # x = F.dropout(x,p=0.6,training=self.training)
        x = self.gat1(x,edge_index)
        # print(x.shape)
        x = F.elu(x)
        # x = F.dropout(x,p=0.6,training=self.training)
        x = self.gat2(x,edge_index)
        
        x = F.elu(x)
        
        x = self.gat3(x,edge_index)
        
        return F.log_softmax(x, dim=1)

    
class VGAT(nn.Module):
    def __init__(self,in_features,n_hidden,n_classes,n_heads,prior_distro,dropout=0.6,leaky_relu=0.2):
        super(VGAT,self).__init__()     
        self.layers = nn.ModuleList([
            VGATLayer(
                in_features,n_hidden,1,prior_distribution=prior_distro,dropout=dropout,leaky_relu=leaky_relu
            ),
            VGATLayer(
                n_hidden*n_heads,n_hidden,1,prior_distribution=prior_distro,dropout=dropout,is_concat=False,leaky_relu=leaky_relu
            ),
            VGATLayer(
                n_hidden*n_heads,n_classes,1,prior_distribution=prior_distro,dropout=dropout,is_concat=False,leaky_relu=leaky_relu
            ),
        ])
    
    def forward(self,data):
        total_kl_divergence = 0
        total_kl_attn = 0
        
        x,edge_index = data.x,data.edge_index
        
        for i,layer in enumerate(self.layers):
            x,kl_divergence,kl_attn = layer(x,edge_index)
            
            if i < len(self.layers) - 1:
                x = F.elu(x)
            
            total_kl_divergence += kl_divergence
            total_kl_attn += kl_attn
            
        x = x.unsqueeze(dim=1)
        
        return x,total_kl_divergence,total_kl_attn