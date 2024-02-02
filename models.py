from typing import *
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.nn.inits import reset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from torch.nn import Parameter, ModuleList, Linear
from torch_geometric.nn import MLP, MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.nn import GINConv
from torch_geometric.nn.conv import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax, degree
import torch_scatter
from entropy import mixed_v_entropy

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros

class QGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, node_centrality=None, edge_centrality=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)

        norm_quantum = self.edge_updater(edge_index, node_centrality=node_centrality, edge_centrality=edge_centrality)

        # degree norm
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_deg = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm_quantum=norm_quantum, norm_deg=norm_deg)

        out += self.bias

        return out
    
    def edge_update(self, node_centrality: Tensor, edge_centrality: Tensor) -> Tensor:
        norm = torch.cat([edge_centrality, node_centrality])
        return norm

    def message(self, x_j, norm_deg, norm_quantum):
        out = norm_deg.view(-1, 1) * norm_quantum.view(-1, 1) * x_j
        return out

class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                num_layers: int, last_layer: str, pool: str, dropout: float, quantum: bool):
        super().__init__()
        self.quantum = quantum
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            if self.quantum:
                self.convs.append(QGCNConv(in_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(in_channels, hidden_channels))
            in_channels = hidden_channels

        self.last_layer = last_layer
        self.pool = pool
        self.dropout = dropout
        if self.last_layer == 'lin':
            self.classify = Linear(hidden_channels, out_channels)
        elif self.last_layer == 'mlp':
            self.classify = MLP([hidden_channels, hidden_channels, out_channels], norm=None, dropout=dropout)
        else:
            raise ValueError("last layer must be 'lin' or 'mlp'.")

    def forward(self, x, edge_index, batch, node_centrality=None, edge_centrality=None):
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.quantum:
                x = conv(x, edge_index, node_centrality, edge_centrality).relu()
            else:
                x = conv(x, edge_index).relu()
        if self.pool == 'add':
            x = global_add_pool(x, batch)
        elif self.pool == 'mean':
            x = global_mean_pool(x, batch)
        else:
            raise ValueError("pool method must me 'add' or 'mean'.")
        
        x = self.classify(x)

        return x

class QGINConv(MessagePassing):
    def __init__(self, nn: Callable, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.register_buffer('eps', torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(0.)

    def forward(self, x: Tensor, edge_index: Adj, node_centrality=None, edge_centrality=None) -> Tensor:
        x = (x, x)
        out = self.propagate(edge_index, x=x, norm_quantum=edge_centrality)
        x_r = x[1]
        out = out * node_centrality.view(-1, 1)
        out = out + (1 + self.eps) * x_r
        return self.nn(out)

    def message(self, x_j: Tensor, norm_quantum) -> Tensor:
        out = x_j * norm_quantum.view(-1, 1)
        return out

class GIN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                num_layers: int, last_layer: str, pool: str, dropout: float, quantum: bool):
        super().__init__()
        self.quantum = quantum
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels]) #, dropout=dropout
            if self.quantum:
                self.convs.append(QGINConv(nn=mlp))
            else:
                self.convs.append(GINConv(nn=mlp))
            in_channels = hidden_channels

        self.last_layer = last_layer
        self.pool = pool
        self.dropout = dropout
        if self.last_layer == 'lin':
            self.classify = Linear(hidden_channels, out_channels)
        elif self.last_layer == 'mlp':
            self.classify = MLP([hidden_channels, hidden_channels, out_channels], norm=None, dropout=dropout)
        else:
            raise ValueError("last layer must be 'lin' or 'mlp'.")

    def forward(self, x, edge_index, batch, node_centrality=None, edge_centrality=None):
        for conv in self.convs:
            if self.quantum:
                x = conv(x, edge_index, node_centrality, edge_centrality).relu()
            else:
                x = conv(x, edge_index).relu()

        if self.pool == 'add':
            x = global_add_pool(x, batch)
        elif self.pool == 'mean':
            x = global_mean_pool(x, batch)
        else:
            raise ValueError("pool method must me 'add' or 'mean'.")
        
        x = self.classify(x)
        
        return x
    
class QGATConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 2, dropout: float = 0.5,
                 add_self_loops: bool = True, **kwargs):
        # may only support heads = 1 now.
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin_src = Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = self.lin_src
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))
        self.bias = Parameter(torch.empty(heads * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, node_centrality: Tensor, edge_centrality: Tensor,
                edge_attr: OptTensor = None, size: Size = None):
        x_src = x_dst = self.lin_src(x).view(-1, self.heads, self.out_channels)
        x = (x_src, x_dst)

        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            num_nodes = x_src.size(0)
            num_nodes = min(num_nodes, x_dst.size(0))
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value='mean', num_nodes=num_nodes)

        alpha, norm_quantum = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr,
                                                node_centrality=node_centrality, edge_centrality=edge_centrality)
        out = self.propagate(edge_index, x=x, alpha=alpha, norm_quantum=norm_quantum, size=size)
        out = out.view(-1, self.heads * self.out_channels)
        out = out + self.bias
        return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    node_centrality: Tensor, edge_centrality: Tensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,size_i: Optional[int]) -> Tensor:
        # sum them up to "emulate" concatenation:
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        norm_quantum = torch.cat([edge_centrality, node_centrality])
        return alpha, norm_quantum

    def message(self, x_j: Tensor, alpha: Tensor, norm_quantum: Tensor) -> Tensor:
        out = alpha.unsqueeze(-1) * x_j
        out = norm_quantum.view(-1, 1, 1) * out
        return out

class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                num_layers: int, last_layer: str, pool: str, dropout: float, quantum: bool):
        super().__init__()
        self.quantum = quantum
        self.heads = 2
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if self.quantum:
                self.convs.append(QGATConv(in_channels if i == 0 else in_channels * self.heads, hidden_channels, heads=self.heads))
            else:
                self.convs.append(GATConv(in_channels if i == 0 else in_channels * self.heads, hidden_channels, heads=self.heads))
            in_channels = hidden_channels

        self.last_layer = last_layer
        self.pool = pool
        self.dropout = dropout
        if self.last_layer == 'lin':
            self.classify = Linear(hidden_channels * self.heads, out_channels)
        elif self.last_layer == 'mlp':
            self.classify = MLP([hidden_channels * self.heads, hidden_channels, out_channels], norm=None, dropout=dropout)
        else:
            raise ValueError("last layer must be 'lin' or 'mlp'.")

    def forward(self, x, edge_index, batch, node_centrality, edge_centrality):
        if self.quantum:
            for conv in self.convs:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = conv(x, edge_index, node_centrality, edge_centrality).relu()
        else:
            for conv in self.convs:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = conv(x, edge_index).relu()

        if self.pool == 'add':
            x = global_add_pool(x, batch)
        elif self.pool == 'mean':
            x = global_mean_pool(x, batch)
        else:
            raise ValueError("pool method must me 'add' or 'mean'.")
        
        x = self.classify(x)

        return x

class QSAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, **kwargs):
        super().__init__('mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l = Linear(in_channels, out_channels, bias=True)
        self.lin_r = Linear(in_channels, out_channels, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.aggr_module.reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, node_centrality: Tensor, edge_centrality: Tensor) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, x=x, edge_centrality=edge_centrality)
        out = self.lin_l(out)

        x_r = x[1]
        x_r = node_centrality.view(-1, 1) * x_r
        out += self.lin_r(x_r)

        return out

    def message(self, x_j: Tensor, edge_centrality: Tensor) -> Tensor:
        out = edge_centrality.view(-1, 1) * x_j
        return out

class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                num_layers: int, last_layer: str, pool: str, dropout: float, quantum: bool):
        super().__init__()
        self.quantum = quantum
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            if self.quantum:
                self.convs.append(QSAGEConv(in_channels, hidden_channels))
            else:
                self.convs.append(SAGEConv(in_channels, hidden_channels))
            in_channels = hidden_channels

        self.last_layer = last_layer
        self.pool = pool
        self.dropout = dropout
        if self.last_layer == 'lin':
            self.classify = Linear(hidden_channels, out_channels)
        elif self.last_layer == 'mlp':
            self.classify = MLP([hidden_channels, hidden_channels, out_channels], norm=None, dropout=dropout)
        else:
            raise ValueError("last layer must be 'lin' or 'mlp'.")

    def forward(self, x, edge_index, batch, node_centrality=None, edge_centrality=None):
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.quantum:
                x = conv(x, edge_index, node_centrality, edge_centrality).relu()
            else:
                x = conv(x, edge_index).relu()
        if self.pool == 'add':
            x = global_add_pool(x, batch)
        elif self.pool == 'mean':
            x = global_mean_pool(x, batch)
        else:
            raise ValueError("pool method must me 'add' or 'mean'.")
        
        x = self.classify(x)

        return x

def get_model(args) -> nn.Module:
    quantum = True if args['model_name'][0] == 'Q' else False
    if args['model_name'] == 'GCN' or args['model_name'] == 'QGCN':
        MODEL = GCN
    elif args['model_name'] == 'GIN' or args['model_name'] == 'QGIN':
        MODEL = GIN
    elif args['model_name'] == 'GAT' or args['model_name'] == 'QGAT':
        MODEL = GAT
    elif args['model_name'] == 'SAGE' or args['model_name'] == 'QSAGE':
        MODEL = SAGE
    model = MODEL(
        in_channels = args['num_features'],
        hidden_channels = args['layer_size'],
        out_channels = args['num_classes'],
        num_layers = args['num_layers'],
        last_layer = args['last_layer'],
        pool = args['pool'],
        dropout = args['dropout'],
        quantum = quantum,
    )
    
    return model

##### entropy for features #####

# class QEGNN(nn.Module):
#     '''
#         QuantumEntropyGNN (entropy for feature attention)
#     '''
#     def __init__(self, input_size, hidden_size, output_size, num_att, dropout) -> None:
#         super().__init__()
#         self.num_att = num_att
#         self.dropout = dropout
#         self.lin1 = torch.nn.Linear(input_size, hidden_size)
#         self.attentions = ModuleList([QEConv(hidden_size) for _ in range(self.num_att)])
#         self.lin2 = torch.nn.Linear(hidden_size, output_size)
    
#     def reset_parameters(self):
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()
#         for i in range(self.num_att):
#             self.attentions[i].reset_parameters()

#     def forward(self, x, edge_index, indices):
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.lin1(x)
#         x = F.relu(x)

#         for i in range(self.num_att):
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             x = self.attentions[i](x, edge_index)
#             x = F.relu(x)

#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = global_mean_pool(x, indices)
#         x = self.lin2(x)
            
#         return x

# class QEConv(MessagePassing):
#     '''
#         Conv layer of QuantumEntropyGNN (entropy for feature attention)
#     '''
#     def __init__(self, hidden_size, requires_grad: bool = True, add_self_loops: bool = True,
#                  **kwargs):
#         super().__init__(**kwargs, aggr='mean')

#         self.requires_grad = requires_grad
#         self.add_self_loops = add_self_loops

#         if requires_grad:
#             self.beta = Parameter(torch.Tensor(1), requires_grad=True)
#         else:
#             self.register_buffer('beta', torch.ones(1))

#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.requires_grad:
#             self.beta.data.fill_(1)

#     def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
#         x_norm = F.normalize(x, p=2., dim=-1)

#         return self.propagate(edge_index, x=x, x_norm=x_norm, size=None)

#     def message(self, x_i: Tensor, x_j: Tensor, x_norm_i: Tensor, x_norm_j: Tensor,
#                 index: Tensor, ptr, size_i) -> Tensor:
#         # entropy similarity
#         l = x_norm_i.size(0)
#         mixed_v_entropy(torch.stack([x_norm_i[0], x_norm_j[0]]).detach(), np.ones(2) / 2)

#         for i in range(l):
#             if (x_norm_i[i] != 0).sum() == 0:
#                 x_norm_i[i][0] += 0.0001
#             if (x_norm_j[i] != 0).sum() == 0:
#                 x_norm_j[i][0] += 0.0001
        
#         sim = [mixed_v_entropy(torch.stack([x_norm_i[i], x_norm_j[i]]).detach(), np.ones(2) / 2) for i in range(l)]
        
#         alpha = torch.tensor(sim).unsqueeze(-1)
#         out = x_j * alpha
#         out = out.float()

#         return out





if __name__ == '__main__':
    ...