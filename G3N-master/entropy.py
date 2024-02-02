import math
from tqdm import tqdm
import numpy as np
import scipy
import torch.nn as nn
import networkx as nx
from typing import *
from qiskit.quantum_info import entropy as v_entropy, DensityMatrix
from torch import Tensor
from torch_geometric.data import Data

def _edgeindex2adj(edge_index: Tensor, num_nodes: int) -> np.ndarray:
    adj = np.zeros([num_nodes, num_nodes], dtype=np.int32)
    for i in range(edge_index.shape[-1]):
        node1 = edge_index[0][i].item()
        node2 = edge_index[1][i].item()
        adj[node1][node2] = 1

    return adj

def _adj2laplacian(adj: np.ndarray) -> np.ndarray:
    degree_matrix = np.diag(np.sum(adj, axis=1))
    laplacian = degree_matrix - adj
    return laplacian

def _get_egonet_idx(adj: List[list], idx: int, hop: int) -> List[list]:
    try:
        G = nx.from_numpy_matrix(np.array(adj))
    except:
        G = nx.DiGraph(np.array(adj))
    ego_graph = nx.ego_graph(G, idx, radius=hop)
    egonet_idx = list(ego_graph.nodes())

    return egonet_idx

def _rm_node(adj: np.ndarray, idx: int) -> np.ndarray:
    if not isinstance(idx, int) and not isinstance(idx, list):
        raise ValueError("Param 'idx' must be int or list")

    # delete target rows and cols
    adj = np.delete(adj, idx, 0)
    adj = np.delete(adj, idx, 1)

    # remove zero-value rows and cols
    adj = adj[~np.all(adj == 0, axis=0)]
    adj = adj.T[~np.all(adj == 0, axis=0)].T
    
    return adj

def _rm_edge(adj: np.ndarray, idx: Tuple[int, int]) -> np.ndarray:
    if not isinstance(idx, tuple):
        raise ValueError("Param 'idx' must be tuple")
    if len(idx) != 2:
        raise ValueError("Tuple 'idx' must have 2 items.")
    
    adj_new = adj.copy()
    # delete target edges
    node0, node1 = idx[0], idx[1]
    adj_new[node0][node1] = 0
    adj_new[node1][node0] = 0

    # remove zero-value rows and cols
    adj_new = adj_new[~np.all(adj_new == 0, axis=0)]
    adj_new = adj_new.T[~np.all(adj_new == 0, axis=0)].T
    
    return adj_new

def vnge(adj: np.ndarray, eig: str = "np") -> float:
    if adj.shape[0] == 0:
        return 0
    
    volumn = adj.sum()
    laplacian = _adj2laplacian(adj)
    if eig == "np":
        eigenvalues, _ = np.linalg.eig(laplacian)
    elif eig == "scipy":
        '''
            Calculate a part of eigenvalues only, not all.
        '''
        eigenvalues, _ = scipy.sparse.linalg.eigs(laplacian.astype('f'), k=5)
    elif eig == 'appro_deg' or eig == 'appro_deg_ge0':
        '''
            Approximate from degree distribution, 
            from the paper "On the Similarity Between von Neumann Graph Entropy and Structural Information: Interpretation, Computation, and Applications"
        '''
        eigenvalues = adj.sum(axis=-1) # degrees as eigenvalues
    else:
        raise ValueError(f'args "eig" must be "np", "scipy", "appro_deg" or ...')
    
    entropy = 0
    for ev in eigenvalues:
        if ev <= 0:
            ev = 1e-17 # the computation inaccuracy may lead positive eigenvalues to be negative
        zz = ev / volumn # norm
        entropy -= zz * np.log2(zz)
    
    return entropy.real

# def cal_node_centrality(data_list: List[Data], hop: int = 0) -> list:
#     '''
#         for sep
#         paper 'Exploring the Node Importance Based on von Neumann Entropy'
#     '''
#     node_centrality = []
#     for graph in tqdm(data_list, desc="Calculating node centrality", ncols=90, leave=False):
#     # for graph in data_list:
#         adj_origin = _edgeindex2adj(graph.edge_index, graph.x.shape[0])
#         entropy_origin = vnge(adj_origin)
        
#         # node centrality
#         entro_list1 = []
#         num_node = adj_origin.shape[0]
#         for idx in range(num_node):
#             ego_idx = _get_egonet_idx(adj_origin, idx, hop=hop)
#             adj = _rm_node(adj_origin, ego_idx)
#             entropy = vnge(adj)
#             entropy_gap = np.abs(entropy - entropy_origin)
#             entro_list1.append(entropy_gap)
#         node_centrality.append(entro_list1)
    
#     return node_centrality

def cal_node_centrality(data_list: List[Data], eig: str, hop: int = 1) -> list:
    '''
        for sep
        paper 'Measuring Vertex Centrality using the Holevo Quantity'
    '''
    node_centrality = []
    for graph in tqdm(data_list, desc="Calculating node centrality", ncols=90, leave=False):
    # for graph in data:
        adj_origin = _edgeindex2adj(graph.edge_index, graph.x.shape[0])
        entropy_origin = vnge(adj_origin, eig=eig)
        
        # node centrality
        entro_list = []
        num_node = adj_origin.shape[0]
        for idx in range(num_node):
            ego_idx = _get_egonet_idx(adj_origin, idx, hop=hop)
            ego_compl_idx = [idx for idx in list(range(num_node)) if idx not in ego_idx]
            ego_size = len(ego_idx)
            ego_compl_size = num_node - ego_size

            adj_ego = _rm_node(adj_origin, ego_compl_idx)
            entropy_ego = vnge(adj_ego, eig=eig)

            adj_compl = _rm_node(adj_origin, ego_idx)
            entropy_compl = vnge(adj_compl, eig=eig)
            entropy_gap = entropy_origin - (ego_compl_size / num_node * entropy_compl + ego_size / num_node * entropy_ego)
            if entropy_gap == 0.0:
                entropy_gap = np.log2(num_node) * 0.75 # any better value?
            if eig == 'appro_deg_ge0':
                entropy_gap = np.abs(entropy_gap)
            entro_list.append(entropy_gap)

        node_centrality.append(entro_list)
    
    return node_centrality

def cal_edge_centrality(data_list, eig: str) -> list:
    '''
        for sep
        paper 'Edge Centrality via the Holevo Quantity'
    '''
    edge_centrality = []
    for graph in tqdm(data_list, desc="Calculating edge centrality", ncols=90, leave=False):
        adj_origin = _edgeindex2adj(graph.edge_index, graph.x.shape[0])
        entropy_origin = vnge(adj_origin, eig=eig)

        # edge centrality
        entro_list = []
        num_edge = graph.edge_index.shape[-1]
        for edge in graph.edge_index.T:
            adj_complement = _rm_edge(adj_origin, (edge[0].item(), edge[1].item()))
            entropy = vnge(adj_complement, eig=eig)
            entropy_gap = entropy - ((num_edge - 1) / num_edge) * entropy_origin # np.abs(entropy - ((num_edge - 1) / num_edge) * entropy_origin)
            if eig == 'appro_deg_ge0':
                entropy_gap = np.abs(entropy_gap)
            entro_list.append(entropy_gap)
        edge_centrality.append(entro_list)
    
    return edge_centrality

# ---------------------------------------------------------------------------------------

def edges_mixed_entropy(data: Data, features: Tensor) -> dict:
    '''
        Calculate the mixed state entropy of features for all edges in a graph.
        Parameters:
            param1: Data obj
            param2: features of the edge
        Return:
            an adacency-list-like dict with value of entropy
    '''

    num_nodes = len(data.x)
    entropy_list = {i:[] for i in range(num_nodes)}
    features = features.detach()

    num_mixed_nodes = data.x.shape[0]
    weights = np.ones(2) / 2 # for edge

    for edge in data.edge_index.T:
        node1 = edge[0].item()
        node2 = edge[1].item()
        feature = [features[node1], features[node2]]
        entropy = mixed_v_entropy(feature, weights)
        entropy_list[node1].append(entropy)
    
    return entropy_list

def mixed_v_entropy(features: list, weights: list = None) -> float:
    ''' 
        Calculate the von Neumann entropy of a list of features.
        Parameters:
            param1: classical features
            param2: weights for mixing pure quantum states, default: 0.
        Returns:
            von Neumann entropy with value of (0, logd)
    '''
    if weights is None:
        l = len(features)
        weights = np.ones([l]) / l

    density_matrix = _mixed_encoding(features, weights)
    entro = v_entropy(density_matrix)

    return entro

def _align_norm_feature(feature: list) -> list:
    '''
        Parameters:
            unnormalized features with arbitrary dim
        Returns:
            normalized features with 2^n dim
    '''
    feature = list(feature)
    upper_bound = np.power(
                        2,
                        math.ceil(np.log2(len(feature)))
                    )
    feat_len = len(feature)
    feature = feature + [0] * (upper_bound - feat_len)
    norm_feat = feature / np.linalg.norm(feature)
    norm_feat = list(norm_feat)
    
    return norm_feat

# def _get_weights(node_idx: list, num_nodes: int, edge_list: list) -> list:
#     '''
#         Parameters:
#             statistic of the graph
#         Returns:
#             weights of nodes
#     '''
#     adj_nodes = []
#     for edge in edge_list:
#         if edge[0] == node_idx:
#             adj_nodes.append(edge[1])
#         elif edge[1] == node_idx:
#             adj_nodes.append(edge[0])
    
#     # TODO: The weight should be calculated by a criterion like node degree.
#     # Here I just use a simple average strategy for every node.
#     weights = np.ones([len(adj_nodes)]) / len(adj_nodes)

#     return weights

def _get_weights() -> list:
    '''
        Just for edge now
        Parameters:
            statistic of the graph
        Returns:
            weights of nodes
    '''
    weights = np.ones(2) / 2

    return weights

def _amplitude_encoding(feature: list) -> list:
    '''
        Parameters:
            param1: features to be encoded
        Return:
            statevector of the pure quantum state
    '''
    initial_state = _align_norm_feature(feature)

    return initial_state

def _mixed_encoding(features: list, weights: list) -> DensityMatrix:
    '''
        Parameters:
            param1: features to be encoded
            param2: the mixing weight for nodes
        Return:
            density matrix of the mixed quantum state
    '''
    assert len(features) == len(weights)
    # assert np.sum(weights) == 1.0

    # encode features into several pure quantum states
    initial_states = [_amplitude_encoding(feature) for feature in features]

    # mix pure quantum states into the mixed quantum state
    density_matrices = [DensityMatrix(initial_state).data for initial_state in initial_states]
    density_matrix = np.zeros_like(density_matrices[0])
    for d, w in zip(density_matrices, weights):
        density_matrix += d * w

    return density_matrix

def _test_mixed_entropy():
    features = np.array([[1.5, 2.0, -3.0, 10.0],
                        [2.5, 2.0, -5.0, 7.0]])
    # features = np.array([[2.0, 1.3],
    #                     [2.0, 1.3]])
    # edge_list = [[0, 1], [1, 2], [2, 0], [0, 0], [1, 1], [2, 2]]

    weights = _get_weights()
    entro = mixed_v_entropy(features, weights)

    print(entro)

if __name__ == '__main__':
    _test_mixed_entropy()