import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn.conv import GCNConv, AGNNConv, GATConv, SGConv
from torch_geometric.nn import APPNP as Appnp
from torch_geometric.utils import add_self_loops, softmax

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class SGC(torch.nn.Module):
    def __init__(self, num_features, num_classes, dropout):
        super().__init__()
        self.conv1 = SGConv(num_features, num_classes, K=2, cached=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x, edge_index = x, edge_index
        x = self.conv1(x, edge_index)
        return x

class AGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, dropout):
        super().__init__()
        self.lin1 = torch.nn.Linear(num_features, 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.lin2(x)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

class GraphSage(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_class):
        super().__init__()
        self.graphsage = GraphSAGE(
                            in_channels=num_features,
                            hidden_channels=hidden_size,
                            out_channels=num_class,
                            num_layers=2,
                        )
    def forward(self, x, edge_index):
        x = self.graphsage(x, edge_index)
        return x

class APPNP(torch.nn.Module):
    def __init__(self, num_feature, hidden_size, num_class, dropout, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = nn.Linear(num_feature, hidden_size)
        self.lin2 = nn.Linear(hidden_size, num_class)
        self.prop1 = Appnp(K, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size1)
        self.lin2 = nn.Linear(hidden_size1, hidden_size2)
        self.lin3 = nn.Linear(hidden_size2, output_size)
        self.dropout = dropout
    
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p = self.dropout)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p = self.dropout)
        x = self.lin3(x)
        return x