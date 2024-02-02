import os
import math
import random
from typing import *
from typing import Any
import numpy as np
import torch
import networkx as nx
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.utils import to_networkx, degree
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.dataset import Dataset
from rawdata_process import get_data
from entropy import cal_node_centrality, cal_edge_centrality

import torch_geometric.transforms as T
from k_gnn import TwoMalkin, ConnectedThreeLocal, ConnectedThreeMalkin

class SepDataset(InMemoryDataset):
    '''
        Dataset for Graph level tasks, separately.
    '''
    def __init__(self, dataset_name, data, pre_transform):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_list: List[Data] = data
        self.pre_transform = pre_transform
        self.pre_filt()
        self.pre_trans()

    def pre_trans(self):
        if self.dataset_name == 'IMDB-BINARY' or self.dataset_name == 'IMDB-MULTI':
            self.pre_transform = [IMDBPreTransform()]
        if not isinstance(self.pre_transform, list):
            self.pre_transform = list(self.pre_transform)
        for trans_call in self.pre_transform:
            for d in self.data_list:
                d = trans_call(d)
        
        self.data, _ = self.collate(self.data_list)
    
    def pre_filt(self):
        if self.dataset_name == 'IMDB-BINARY' or self.dataset_name == 'IMDB-MULTI':
            self.filter = lambda x: x.num_nodes <= 70
        elif self.dataset_name == 'PTC_MR':
            self.filter = lambda x: x.num_nodes > 3
        elif self.dataset_name == 'PROTEINS':
            self.filter = lambda x: not (x.num_nodes == 7 and x.num_edges == 12) and x.num_nodes < 450
        elif self.dataset_name == 'NCI1':
            self.filter = lambda x: x.num_nodes > 5
        elif self.dataset_name == 'ENZYMES':
            self.filter = lambda x: x.num_nodes > 3 and x.num_nodes != 100
        else:
            self.filter = lambda x: True
            
        self.filter_mask = [self.filter(d) for d in self.data_list]
        self.data_list = [d for d in self.data_list if self.filter(d)]
    
    def iso_onehot(self):
        self.data.iso_type_2 = torch.unique(self.data.iso_type_2, True, True)[1]
        num_i_2 = self.data.iso_type_2.max().item() + 1
        self.data.iso_type_2 = F.one_hot(
            self.data.iso_type_2, num_classes=num_i_2).to(torch.float)

        self.data.iso_type_3 = torch.unique(self.data.iso_type_3, True, True)[1]
        num_i_3 = self.data.iso_type_3.max().item() + 1
        self.data.iso_type_3 = F.one_hot(
            self.data.iso_type_3, num_classes=num_i_3).to(torch.float)
        
        ''' Do not transfer to comment !!! not simply check !!! '''
        count2, count3 = 0, 0
        for d in self.data_list:
            l2 = d.iso_type_2.shape[0]
            l3 = d.iso_type_3.shape[0]
            d.iso_type_2 = self.data.iso_type_2[count2: count2+l2, :]
            d.iso_type_3 = self.data.iso_type_3[count3: count3+l3, :]
            count2 += l2
            count3 += l3
        assert count2 == self.data.iso_type_2.shape[0]
        assert count3 == self.data.iso_type_3.shape[0]
        return num_i_2, num_i_3
    
    def get(self, idx: int or slice) -> Data:
        return self.data_list[idx]
    
    def __getitem__(self, idx: int or List[bool]) -> Dataset:
        if isinstance(idx, int) or isinstance(idx, slice):
            out = self.get(idx)
        elif isinstance(idx, list) and all(isinstance(item, bool) for item in idx): # for bool masks
            if len(idx) != self.len():
                raise ValueError("The length of bool list 'idx' is different to that of dataset")
            
            out = []
            for i, item in enumerate(idx):
                if item:
                    out.append(self.data_list[i])
        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            out = []
            for i, item in enumerate(idx):
                if item.item():
                    out.append(self.data_list[i])
        elif idx is None:
            out = []
        else:
            if not isinstance(idx, list):
                raise ValueError(f"The index list of SepDataset must be slice or bool list.")
        
        return out

    def len(self):
        return len(self.data_list)
    
    def shuffle(self) -> None:
        random.shuffle(self.data_list)
    
    def to(self, device) -> None:
        for d in self.data_list:
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
        ret = self.data_list[0].x.shape[-1]
        return ret
    
    @property
    def num_classes(self) -> int:
        if self.data_list[0].y.dtype == torch.float32: # regression dataset has no num_class, but return 1 here for model build
            ret = 1
        else:
            label_l = [d.y.item() for d in self.data_list]
            ret = np.array(label_l).max() + 1
        return ret
    
    @property
    def avg_num_nodes(self) -> int:
        num_nodes_l = [d.x.shape[0] for d in self.data_list]
        ret = np.array(num_nodes_l).mean()
        return ret
    
    @property
    def max_num_nodes(self) -> int:
        num_nodes_l = [d.x.shape[0] for d in self.data_list]
        ret = np.array(num_nodes_l).max()
        return ret
    
    @property
    def avg_num_edges(self) -> int:
        num_edges_l = [d.edge_index.shape[-1] / 2 for d in self.data_list]
        ret = np.array(num_edges_l).mean()
        return ret
    
    @property
    def max_num_edges(self) -> int:
        num_edges_l = [d.edge_index.shape[-1] / 2 for d in self.data_list]
        ret = np.array(num_edges_l).max()
        return ret
    
    @property
    def avg_degree(self) -> int:
        avg_deg_l = []
        for graph in self.data_list:
            graph = to_networkx(graph)
            avg_deg = sum(dict(graph.degree()).values()) / len(graph) / 2
            avg_deg_l.append(avg_deg)
        ret = np.array(avg_deg_l).mean()
        return ret
    
    @property
    def avg_cluster(self) -> int:
        avg_cluster_l = []
        for graph in self.data_list:
            graph = to_networkx(graph)
            clustering_coefficient = nx.average_clustering(graph)
            avg_cluster_l.append(clustering_coefficient)
        ret = np.array(avg_cluster_l).mean()
        return ret
    
    @property
    def avg_diameter(self) -> int:
        avg_diameter_l = []
        for graph in self.data_list:
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

def get_dataset(dataset_name: str, recal: bool, cal_entro: bool, nc_norm: int = 1, ec_norm: int = 10, eig: str = "appro_deg") -> SepDataset:
    data_list = get_data(dataset_name, recal)
    
    dataset = SepDataset(
        dataset_name,
        data_list,
        pre_transform=[TwoMalkin(), ConnectedThreeLocal()])
    
    if cal_entro:
        # 1-order
        obj_file = f'../datasets/obj/{dataset_name}_nc_{eig}.pt'
        if not recal and os.path.exists(obj_file):
            node_centralities = torch.load(obj_file)
        else:
            node_centralities = cal_node_centrality(dataset, order=1, eig=eig)
            torch.save(node_centralities, obj_file)

        obj_file = f'../datasets/obj/{dataset_name}_ec_{eig}.pt'
        if not recal and os.path.exists(obj_file):
            edge_centralities = torch.load(obj_file)
        else:
            edge_centralities = cal_edge_centrality(dataset, order=1, eig=eig)
            torch.save(edge_centralities, obj_file)
        
        node_centralities = [nc for nc, mask in zip(node_centralities, dataset.filter_mask) if mask]
        edge_centralities = [nc for nc, mask in zip(edge_centralities, dataset.filter_mask) if mask]
        
        for graph, nc, ec in zip(dataset, node_centralities, edge_centralities):
            nc = torch.tensor(nc).float() * nc_norm
            ec = torch.tensor(ec).float() * ec_norm
            if nc_norm == -1.0:
                nc.fill_(1.0)
            if ec_norm == -1.0:
                ec.fill_(1.0)
            
            graph.node_centrality1 = nc
            graph.edge_centrality1 = ec
            
        # 2-order
        # obj_file = f'../datasets/k-gnn/{dataset_name}_nc2_{eig}.pt'
        # if not recal and os.path.exists(obj_file):
        #     node_centralities = torch.load(obj_file)
        # else:
        #     node_centralities = cal_node_centrality(dataset, order=2, eig=eig)
        #     torch.save(node_centralities, obj_file)

        # obj_file = f'../datasets/k-gnn/{dataset_name}_ec2_{eig}.pt'
        # if not recal and os.path.exists(obj_file):
        #     edge_centralities = torch.load(obj_file)
        # else:
        #     edge_centralities = cal_edge_centrality(dataset, order=2, eig=eig)
        #     torch.save(edge_centralities, obj_file)
        
        # for graph, nc, ec in zip(dataset, node_centralities, edge_centralities):
        #     nc = torch.tensor(nc).float() * nc_norm
        #     ec = torch.tensor(ec).float() * ec_norm
        #     if nc_norm == -1.0:
        #         nc.fill_(1.0)
        #     if ec_norm == -1.0:
        #         ec.fill_(1.0)
            
        #     graph.node_centrality2 = nc
        #     graph.edge_centrality2 = ec
            
        # 3-order
        # obj_file = f'../datasets/k-gnn/{dataset_name}_nc3_{eig}.pt'
        # if not recal and os.path.exists(obj_file):
        #     node_centralities = torch.load(obj_file)
        # else:
        #     node_centralities = cal_node_centrality(dataset, order=3, eig=eig)
        #     torch.save(node_centralities, obj_file)

        # obj_file = f'../datasets/k-gnn/{dataset_name}_ec3_{eig}.pt'
        # if not recal and os.path.exists(obj_file):
        #     edge_centralities = torch.load(obj_file)
        # else:
        #     edge_centralities = cal_edge_centrality(dataset, order=3, eig=eig)
        #     torch.save(edge_centralities, obj_file)
        
        # for graph, nc, ec in zip(dataset, node_centralities, edge_centralities):
        #     nc = torch.tensor(nc).float() * nc_norm
        #     ec = torch.tensor(ec).float() * ec_norm
        #     if nc_norm == -1.0:
        #         nc.fill_(1.0)
        #     if ec_norm == -1.0:
        #         ec.fill_(1.0)
            
        #     graph.node_centrality3 = nc
        #     graph.edge_centrality3 = ec
    
    return dataset

class IMDBPreTransform(object):
    def __call__(self, data):
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)
        data = TwoMalkin()(data)
        data = ConnectedThreeMalkin()(data)
        data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        data.x = F.one_hot(data.x, num_classes=136).to(torch.float)
        return data

class IMDBFilter:
    def __call__(self, data):
        return data.num_nodes <= 70

def dense2sparse(dense_tensor: torch.Tensor) -> torch.Tensor:
    '''
        Convert dense (strided) tensor to sparse tensor.
    '''
    indices = torch.nonzero(dense_tensor)
    rows = indices[:, 0]
    cols = indices[:, 1]
    values = dense_tensor[rows, cols]
    sparse_tensor = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, dense_tensor.size())
    
    return sparse_tensor

def sparse2dense(sparse_tensor: torch.Tensor) -> torch.Tensor:
    '''
        Convert sparse tensor to dense (strided) tensor.
    '''
    dense_tensor = torch.zeros(sparse_tensor.size()).to(sparse_tensor.device)
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    dense_tensor[indices[0], indices[1]] = values
    
    return dense_tensor
if __name__ == '__main__':
    ...