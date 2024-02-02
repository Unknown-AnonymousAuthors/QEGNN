import os
import csv
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from typing import *
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset, QM9, ZINC
from torch_geometric.utils import to_undirected
import networkx as nx
# from ogb.graphproppred import PygGraphPropPredDataset

def get_data(dataset_name: str, recal=False) -> List[Data]:
    '''
        Get torch_geometric.data.Dataset obj from disk or recalculated from raw data
        Parameters:
            param1: Data obj
            param2: if recalculate the Data obj
        Return:
            off-the-shelf Data obj or recalculated Data obj
    '''
    obj_file = f'../datasets/obj/{dataset_name}.pt'
    if not recal and os.path.exists(obj_file):
        data_list = torch.load(obj_file) # maybe not exist
        return data_list
    
    # recal the dataset from raw data
    if dataset_name in ['MUTAG', 'PROTEINS', 'NCI1', 'NCI109', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI', 'PTC_MR'] + ['COLLAB', 'DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        dataset = TUDataset('./datasets/raw/', dataset_name)
    # elif dataset_name in ['ogbg-molhiv']: # 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-ppa', 'ogbg-code2'
    #     dataset = PygGraphPropPredDataset(root='./datasets/raw/', name=dataset_name)
    #     mask_dic = dataset.get_idx_split()
    #     train_idx = mask_dic['train']
    #     test_idx = mask_dic['test']
    #     val_idx = mask_dic['valid']
    #     dim = train_idx.max() + 1
    #     train_mask = [False] * dim
    #     test_mask = [False] * dim
    #     val_mask = [False] * dim
    #     for idx in train_idx:
    #         train_mask[idx.item()] = True
    #     for idx in test_idx:
    #         test_mask[idx.item()] = True
    #     for idx in val_idx:
    #         val_mask[idx.item()] = True
    #     torch.save((train_mask, test_mask, val_mask), f"./datasets/obj/{dataset_name}_masks.pt")
    # elif dataset_name in ['EXP']:
    #     with open('./datasets/raw/EXP/raw/EXP.pkl', 'rb') as file:
    #         dataset = pickle.load(file)
    # elif dataset_name in ['SR']:
    #     dataset = SRDataset('./datasets/raw/SR')
    # elif dataset_name in ['CSL']:
    #     dataset = GNNBenchmarkDataset('./datasets/raw/', dataset_name)
    # elif dataset_name in ['QM9']:
    #     dataset = QM9(f'./datasets/raw/QM9')
    # elif dataset_name in ['ZINC_full', 'ZINC_subset']:
    #     subset = True if dataset_name == 'ZINC_subset' else False
    #     # ZINC_train = TUDataset('./datasets/raw/', 'ZINC_train')
    #     # ZINC_test = TUDataset('./datasets/raw/', 'ZINC_test')
    #     # ZINC_val = TUDataset('./datasets/raw/', 'ZINC_val')
    #     # dataset = ZINC_train + ZINC_test + ZINC_val
    #     # train_mask = [True] * 220011 + [False] * (249456 - 220011)
    #     # test_mask = [False] * 220011 + [True] * (5000) + [False] * (249456 - 220011 - 5000)
    #     # val_mask = [False] * (220011 + 5000) + [True] * (249456 - 220011 - 5000)
    #     ZINC_train = ZINC('./datasets/raw/ZINC/', subset=subset, split='train')
    #     ZINC_test = ZINC('./datasets/raw/ZINC/', subset=subset, split='test')
    #     ZINC_val = ZINC('./datasets/raw/ZINC/', subset=subset, split='val')
    #     dataset = ZINC_train + ZINC_test + ZINC_val
        # len_train, len_test, len_val = len(ZINC_train), len(ZINC_test), len(ZINC_val)
        # train_mask = [True] * len_train + [False] * (len_test + len_val)
        # test_mask = [False] * len_train + [True] * (len_test) + [False] * (len_val)
        # val_mask = [False] * (len_train + len_test) + [True] * (len_val)
        # torch.save((train_mask, test_mask, val_mask), f"./datasets/obj/{dataset_name}_masks.pt")
    else:
        raise ValueError(f"The input name of dataset '{dataset_name}' is invalid.")
    
    if dataset_name in ['ZINC_full', 'ZINC_subset', 'QM9']:
        task_type = 'regression'
    else:
        task_type = 'classification'
    
    data_list = list(dataset)
    _rectify_data(data_list, task_type)
    _verify_data(data_list, task_type)
    
    if dataset_name in ['REDDIT-BINARY', 'REDDIT-MULTI-5K']: # for extremely sparse features, saving storage space
        for data in data_list:
            data.x = dense2sparse(data.x)
    torch.save(data_list, obj_file)

    return data_list

def _rectify_data(data_list: List[Data], task_type: str) -> None:
    # fix begin value of edge_index with 0
    for data in data_list:        
        m_ei = data.edge_index.min()
        if m_ei != 0:
            new_ei = data.edge_index - m_ei
            data.edge_index = new_ei
    
    # add feature
    if not hasattr(data_list[0], 'x') or data_list[0].x is None:
        num_nodes_l = []
        for data in data_list:
            num_nodes = data.edge_index.max() + 1
            num_nodes_l.append(num_nodes)
        dim = np.array(num_nodes_l).max()
        
        for num_nodes, data in zip(num_nodes_l, data_list):
            feature = torch.zeros(num_nodes, dim).float()
            for i in range(num_nodes):
                feature[i, i] = 1.0
            data.x = feature
    
    # ensure input feature is torch.float32 rather than torch.long
    if data_list[0].x.dtype != torch.float32:
        for data in data_list:
            data.x = data.x.float()
            
    # ensure dtype of label
    if task_type == 'classification' and data_list[0].y.dtype != torch.int64:
        for data in data_list:
            data.y = data.y.long()
    if task_type == 'regression' and data_list[0].y.dtype != torch.float32:
        for data in data_list:
            data.y = data.y.float()
    
    # delete useless tensors (only for undirected graphs without edge attributes)
    if hasattr(data_list[0], 'edge_attr'):
        for data in data_list:
            if hasattr(data, 'edge_attr'):
                del data['edge_attr']
    
    # rectify labels shape
    if data_list[0].y.dim() == 2:
        for data in data_list:
            data.y = data.y.squeeze(0)
    
    # add one-hot feature
    if data_list[0].x is None:
        max_idx_l = [data.num_nodes for data in data_list]
        max_graph_size = np.array(max_idx_l).max()
        for data in data_list:
            x_l = []
            for idx in range(data.num_nodes):
                feature = torch.tensor([0.] * idx + [1.] + [0.] * (max_graph_size - 1 - idx))
                x_l.append(feature)
            data.x = torch.stack(x_l)

def _verify_data(data_list: List[Data], task_type: str) -> None:
    '''
        verify the property of data.Data() after processing raw data from raw files
    '''

    for data in tqdm(data_list, desc="Verifying dataset", ncols=80, leave=False):
    # for data in data_list:
        feature = data.x
        labels = data.y
        edge_index = data.edge_index

        # 1.verify feature
        assert isinstance(feature, torch.Tensor)
        assert feature.dim() == 2
        assert feature.dtype == torch.float32

        # 2.verify label
        assert isinstance(labels, torch.Tensor)
        assert labels.dim() == 1
        if task_type == 'classification':
            assert labels.dtype == torch.int64
            assert (labels < 0).sum().item() == 0 # label value begin with 0
        else:
            assert labels.dtype == torch.float32

        # 3.verify edge_index
        assert isinstance(edge_index, torch.Tensor)
        assert edge_index.dim() == 2
        assert edge_index.min() == 0 # node id in edge index begin with 0

        # 4.adj symmetric
        num_edges = edge_index.shape[1]
        num_nodes = data.num_nodes
        count1, count2 = 0, 0
        edge_dict = {i:[] for i in range(num_nodes)} # use adj list for fast varifying
        edge_index = edge_index.T.numpy().tolist()
        for edge in edge_index:
            if edge[1] not in edge_dict[edge[0]]:
                edge_dict[edge[0]].append(edge[1])
        edge_index_reverse = [[edge[1], edge[0]] for edge in edge_index]
        for edge_reverse in edge_index_reverse:
            if edge_reverse[1] in edge_dict[edge_reverse[0]]: # check asymmetric
                count1 += 1
            if edge_reverse[0] == edge_reverse[1]: # check self loop
                count2 += 1
        assert count1 == num_edges # all edges have their reverse
        assert count2 == 0         # no self-loops

def _edge_symmetric(edge_index: list, num_nodes: int) -> list:
    '''
        Make the adj matrix (edge list) symmetric without self loops
        Parameters:
            param1: unsymmetric edge list
            param2: number of nodes
        Return:
            symmetric edge list
    '''
    if edge_index.shape[0] != 2 or len(edge_index.numpy().shape) != 2:
        raise ValueError('the shape of edge_index should be [2, len]')
    edge_index = edge_index.T.numpy().tolist()

    # delete self loop
    edge_index = [edge for edge in edge_index if edge[0]!=edge[1]]
    
    # symmetric
    edge_dict = {i:[] for i in range(num_nodes)}
    for edge in edge_index:
        if edge[1] not in edge_dict[edge[0]]:
            edge_dict[edge[0]].append(edge[1])
    edge_index_reverse = [[edge[1], edge[0]] for edge in edge_index]
    for edge_reverse in edge_index_reverse:
        if edge_reverse[1] not in edge_dict[edge_reverse[0]]:
            edge_index.append(edge_reverse)

    edge_index = torch.tensor(edge_index).T
    return edge_index

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

class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  #sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i, datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# class SepTUDataset(Dataset):
#     ''' Separate Every Graph '''
#     def __init__(self, root, dataset_name, transform=None, pre_transform=None):
#         if dataset_name not in ['MUTAG']:
#             raise ValueError('The name of dataset is not valid.')
#         self.dataset_name = dataset_name

#         super(SepTUDataset, self).__init__(root, transform, pre_transform)

#         self.process()

#     @property
#     def raw_file_names(self):
#         if self.dataset_name == 'MUTAG':
#             return [
#                 f'MUTAG_A.txt',
#                 f'MUTAG_graph_indicator.txt',
#                 f'MUTAG_graph_labels.txt',
#                 f'MUTAG_node_labels.txt'
#             ]

#     @property
#     def processed_file_names(self):
#         if self.dataset_name == 'MUTAG':
#             return ['mutag_data.pt']

#     def process(self):
#         # Load raw data from files.
#         with open(os.path.join(self.raw_dir, f'{self.dataset_name}_A.txt'), 'r') as f:
#             edges = [tuple(map(int, line.strip().split(','))) for line in f]
#             edges = [(v1-1, v2-1) for (v1, v2) in edges]
#         with open(os.path.join(self.raw_dir, f'{self.dataset_name}_graph_indicator.txt'), 'r') as f:
#             graph_ids = [int(line.strip()) - 1 for line in f]
#         with open(os.path.join(self.raw_dir, f'{self.dataset_name}_graph_labels.txt'), 'r') as f:
#             graph_labels = [int(line.strip()) for line in f]
#             graph_labels = [label if label == 1 else 0 for label in graph_labels]
#         with open(os.path.join(self.raw_dir, f'{self.dataset_name}_node_labels.txt'), 'r') as f:
#             node_labels = [int(line.strip()) for line in f]

#         # Process data into PyTorch geometric format.
#         num_nodes = max(graph_ids) + 1
#         data_list = []

#         for graph_id in range(num_nodes):
#             node_indices = (np.array(graph_ids) == graph_id).nonzero()[0]
#             low_ind, up_ind = node_indices[0], node_indices[-1]
#             edge_indices = [
#                 [i - node_indices[0], j - node_indices[0]]
#                 for i, j in edges
#                 if i >= low_ind and i <= up_ind
#                 and j >= low_ind and j <= up_ind
#                 # and i < j
#             ]
#             edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
#             x = torch.zeros((len(node_indices), max(node_labels) + 1))
#             for i, node in enumerate(node_indices):
#                 x[i][node_labels[node]] = 1
#             y = torch.tensor([graph_labels[graph_id]], dtype=torch.long)
#             data = Data(x=x, edge_index=edge_indices, y=y)
#             data_list.append(data)

#         self.data = data_list

#     def len(self):
#         return len(self.data)

#     def get(self, idx):
#         data = self.data[idx]
#         return data

if __name__ == "__main__":
    ...