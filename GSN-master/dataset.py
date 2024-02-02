import os
import math
import random
from typing import *
import numpy as np
import torch
import networkx as nx
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Dataset
from torch_geometric.data.dataset import Dataset
from rawdata_process import get_data
from entropy import cal_node_centrality, cal_edge_centrality

class SepDataset(Dataset):
    '''
        Dataset for Graph level tasks, separately.
    '''
    def __init__(self, data):
        super().__init__()
        self.data: List[Data] = data

    def get(self, idx: int or slice) -> Data:
        return self.data[idx]
    
    def __getitem__(self, idx: int or List[bool]) -> Dataset:
        if isinstance(idx, int) or isinstance(idx, slice):
            out = self.get(idx)
        elif isinstance(idx, list) and all(isinstance(item, bool) for item in idx): # for bool masks
            if len(idx) != self.len():
                raise ValueError("The length of bool list 'idx' is different to that of dataset")
            
            out = []
            for i, item in enumerate(idx):
                if item:
                    out.append(self.data[i])
        elif idx is None:
            out = []
        else:
            if not isinstance(idx, list):
                raise ValueError(f"The index list of SepDataset must be slice or bool list.")
        
        return out

    def len(self):
        return len(self.data)
    
    def shuffle(self) -> None:
        random.shuffle(self.data)
    
    def to(self, device) -> None:
        for d in self.data:
            d.to(device=device)

    # TODO
    def statistic_info(self, mode: str = 'simple') -> Dict:
        if mode == 'simple':
            ret = {
                'num_features': self.num_features,
                'num_classes': self.num_classes,
            }
        elif mode == 'middle':
            ret = {
                'num_features': self.num_features,
                'num_classes': self.num_classes,
                'num_graphs': len(self),
                'avg_num_nodes': self.avg_num_nodes,
                'avg_num_edges': self.avg_num_edges,
                'avg_node_deg': self.avg_degree,
            }
        elif mode == 'complex':
            ret = {
                'num_features': self.num_features,
                'num_classes': self.num_classes,
                'num_graphs': len(self),
                'avg_num_nodes': self.avg_num_nodes,
                'avg_num_edges': self.avg_num_edges,
                'avg_node_deg': self.avg_degree,
                'avg_cluster': self.avg_cluster,
                'avg_diameter': self.avg_diameter,
                'max_num_nodes': self.max_num_nodes,
                'max_num_edges': self.max_num_edges,
            }
        else:
            raise ValueError('mode must be simple/middle/complex.')
        return ret
    
    @property
    def num_features(self) -> int:
        ret = self.data[0].x.shape[-1]
        return ret
    
    @property
    def num_classes(self) -> int:
        if self.data[0].y.dtype == torch.float32: # regression dataset has no num_class, but return 1 here for model build
            ret = 1
        else:
            label_l = [d.y.item() for d in self.data]
            ret = np.array(label_l).max() + 1
        return ret
    
    @property
    def avg_num_nodes(self) -> int:
        num_nodes_l = [d.x.shape[0] for d in self.data]
        ret = np.array(num_nodes_l).mean()
        return ret
    
    @property
    def max_num_nodes(self) -> int:
        num_nodes_l = [d.x.shape[0] for d in self.data]
        ret = np.array(num_nodes_l).max()
        return ret
    
    @property
    def avg_num_edges(self) -> int:
        num_edges_l = [d.edge_index.shape[-1] / 2 for d in self.data]
        ret = np.array(num_edges_l).mean()
        return ret
    
    @property
    def max_num_edges(self) -> int:
        num_edges_l = [d.edge_index.shape[-1] / 2 for d in self.data]
        ret = np.array(num_edges_l).max()
        return ret
    
    @property
    def avg_degree(self) -> int:
        avg_deg_l = []
        for graph in self.data:
            graph = to_networkx(graph)
            avg_deg = sum(dict(graph.degree()).values()) / len(graph) / 2
            avg_deg_l.append(avg_deg)
        ret = np.array(avg_deg_l).mean()
        return ret
    
    @property
    def avg_cluster(self) -> int:
        avg_cluster_l = []
        for graph in self.data:
            graph = to_networkx(graph)
            clustering_coefficient = nx.average_clustering(graph)
            avg_cluster_l.append(clustering_coefficient)
        ret = np.array(avg_cluster_l).mean()
        return ret
    
    @property
    def avg_diameter(self) -> int:
        avg_diameter_l = []
        for graph in self.data:
            graph = to_networkx(graph)
            try:
                diameter = nx.diameter(graph)
            except:
                diameter = 0
            avg_diameter_l.append(diameter)
        ret = np.array(avg_diameter_l).mean()
        return ret

class KFoldIter():
    def __init__(self, num_data, folds=10) -> None:
        self.num_data = num_data
        self.folds = folds
    
    def __iter__(self):
        self.cursor = 0
        self.pos = [math.floor(fold * self.num_data / self.folds) for fold in range(self.folds + 1)]

        return self
    
    def __next__(self) -> tuple:
        if self.cursor == self.folds:
            raise StopIteration
    
        pos1 = self.pos[self.cursor]
        pos2 = self.pos[self.cursor + 1]
        train_mask = [True] * pos1 + [False] * (pos2-pos1) + [True] * (self.num_data - pos2)
        test_mask = [not m for m in train_mask]

        self.cursor += 1

        return train_mask, test_mask

def get_dataset(dataset: str, recal: bool, cal_entro: bool, nc_norm: int = 1, ec_norm: int = 1, eig: str = "appro_deg_ge0") -> SepDataset:
    data_list = get_data(dataset, recal)
    
    if cal_entro:
        obj_file = f'../datasets/obj/{dataset}_nc_{eig}.pt'
        if not recal and os.path.exists(obj_file):
            node_centralities = torch.load(obj_file)
        else:
            node_centralities = cal_node_centrality(data_list, eig=eig)
            torch.save(node_centralities, obj_file)

        obj_file = f'../datasets/obj/{dataset}_ec_{eig}.pt'
        if not recal and os.path.exists(obj_file):
            edge_centralities = torch.load(obj_file)
        else:
            edge_centralities = cal_edge_centrality(data_list, eig=eig)
            torch.save(edge_centralities, obj_file)

        for graph, nc, ec in zip(data_list, node_centralities, edge_centralities):
            nc = torch.tensor(nc).float() * nc_norm
            ec = torch.tensor(ec).float() * ec_norm
            if nc_norm == -1.0:
                nc.fill_(1.0)
            if ec_norm == -1.0:
                ec.fill_(1.0)
            graph.node_centrality = nc
            graph.edge_centrality = ec

    dataset = SepDataset(data_list)

    return dataset

if __name__ == '__main__':
    ...