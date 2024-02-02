import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import normalize

# from torch_geometric.datasets import Planetoid
# from torch_geometric.nn import Node2Vec

from entropy import mixed_v_entropy

def edges_mixed_entropy(data: Data, features: Tensor):
    '''
        Calculate the mixed state entropy of all edges.
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

class RandomWalkerSampler:
    ''' Sampler with random walk. Positive sampling by random walk, while negative sampling by random choice. '''
    def __init__(
            self,
            model: torch.nn.Module,
            data: Data,
            walk_length: int = 20,
            window_size: int = 10,
            num_rw_per_node: int = 10,
            prob_mode: str = 'uniform',
        ) -> None:
        self.model = model
        self.data = data
        self.adj_list = self._get_adj_list(self.data)
        self.walk_length = walk_length
        self.window_size = window_size
        self.num_rw_per_node = num_rw_per_node
        self.prob_mode = prob_mode

        if self.prob_mode not in ['uniform', 'node2vec', 'entropy']:
            raise ValueError('The mode of probability for random walk is invalid.')
        if self.prob_mode == 'entropy':
            self.entropy_list = self._get_entropy_list()
    
    def sample_all_nodes(self) -> Tensor:
        '''
            Sample positive sequences and negative sequences for all nodes.
            Return:
                node ids
        '''
        num_nodes = len(self.data.x)
        pos_samples_all = []
        neg_samples_all = []

        for node_idx in range(num_nodes):
            for _ in range(self.num_rw_per_node):
                pos_samples, neg_samples = self.sample_one_node(node_idx)
                pos_samples_all.append(np.array(pos_samples))
                neg_samples_all.append(np.array(neg_samples))
        
        pos_samples_all = torch.tensor(np.array(pos_samples_all))
        neg_samples_all = torch.tensor(np.array(neg_samples_all))
        
        return pos_samples_all, neg_samples_all
    
    def sample_one_node(self, start_node: int) -> Tensor:
        '''
            Sample positive sequences and negative sequences for single node.
            Return:
                node ids
        '''
        pos_content = self.pos_sample(start_node)
        neg_content = self.negative_sample(start_node)

        pos_samples = self._truncate_sequence(pos_content)
        neg_samples = self._truncate_sequence(neg_content)

        return pos_samples, neg_samples
    
    def pos_sample(self, start_node: int) -> Tensor:
        '''
            Positive sampling by random walk.
            Parameters:
                param1: Data obj
                param2: the begining node of the walk
                param3: the window size of the node sequence, length = window_size * 2 + 1
            Return:
                node ids
        '''
        if start_node >= len(self.data.x):
            raise ValueError('Index of the start node should always be smaller than the number of nodes.')
        
        pos_content = [start_node]
        for _ in range(self.walk_length - 1):
            adj_nodes = self.adj_list[pos_content[-1]]
            p = self._get_prob_distribution(pos_content[-1], pos_content)
            next_step = np.random.choice(adj_nodes, 1, p=p)[0] # '[0] is for shape
            pos_content.append(next_step)

        pos_content = torch.tensor(pos_content)
        return pos_content
    
    def negative_sample(self, start_node: int) -> Tensor:
        '''
            Negative sampling with random strategy, and every two neighbors don't have an edge.
            Return:
                node ids
        '''
        if start_node >= len(self.data.x):
            raise ValueError('Index of the start node should always be smaller than the number of nodes.')
        
        sequence = range(len(self.data.x[0]))
        neg_content = [start_node]
        while True:
            neg_node = np.random.choice(sequence, 1, replace=False).item()
            if neg_node not in self.adj_list[neg_content[-1]] and neg_node != neg_content[-1]:
                neg_content.append(neg_node)
                
            if len(neg_content) >= self.walk_length:
                break

        neg_content = torch.tensor(neg_content)
        return neg_content

    def _get_prob_distribution(self, current_node: int, sampled: list) -> Tensor:
        ''' Get probability distribution for every step of random walk '''
        if self.prob_mode == 'uniform':
            return None
        elif self.prob_mode == 'node2vec':
            # use sampled
            return None
        elif self.prob_mode == 'entropy':
            prob_distribution = np.array([self.entropy_list[current_node]]) # '[...]' is for shape
            prob_distribution = normalize(prob_distribution, norm='l1')[0]  # '...[0]' is for shape
            
            return prob_distribution
    
    def _get_entropy_list(self):
        if False: # skip-gram by node ids
            node_ids = torch.arange(len(self.data.x))
            features = self.model(node_ids)
        else:    # attributed NN by node features
            features = self.model(self.data.x)
        
        entropy_list = edges_mixed_entropy(self.data, features)

        return entropy_list
    
    def _truncate_sequence(self, node_sequence: Tensor) -> Tensor:
        '''
            Truncate long sequence into shorter ones with a window size.
            Return:
                node ids
        '''
        if not isinstance(node_sequence, Tensor):
            node_sequence = torch.tensor(node_sequence)

        walks = []
        num_sub_sequences = self.walk_length - self.window_size + 1
        for i in range(num_sub_sequences):
            walks.append(np.array(node_sequence[i:i + self.window_size]))
            
        walks = torch.tensor(np.array(walks))
        return walks

    def _get_adj_list(self, data: Data) -> dict:
        '''
            Turn the edge list into adjacency list, which is indexdx by node ids.
            Parameters:
                param1: Data obj
            Return:
                a python dict of nodes adjacency list
        '''
        num_nodes = len(data.x)
        adj_list = {i:[] for i in range(num_nodes)}

        for edge in data.edge_index.T:
            adj_list[edge[0].item()].append(edge[1].item())

        return adj_list

if __name__ == '__main__':
    ...