import torch
import torch.nn as nn
from torch_geometric.utils import degree, is_undirected

from models_misc import mlp
from utils_graph_learning import central_encoder

class GSN_sparse(nn.Module):
    
    def __init__(self,
                 d_in,
                 d_id,
                 d_degree,
                 degree_as_tag,
                 retain_features,
                 id_scope,
                 d_msg,
                 d_up,
                 d_h,
                 seed,
                 activation_name,
                 bn,
                 aggr='add',
                 msg_kind='general',
                 eps = 0,
                 train_eps = False,
                 flow='source_to_target',
                 **kwargs):

        super(GSN_sparse, self).__init__()
        

        d_msg = d_in if d_msg is None else d_msg

        self.flow = flow
        self.aggr = aggr
        self.msg_kind = msg_kind
        self.id_scope = id_scope
        
        self.degree_as_tag = degree_as_tag
        self.retain_features = retain_features

        if degree_as_tag:
            d_in = d_in + d_degree if retain_features else d_degree

        # different GSN variants: extended gin and general formulation (see supplementary material)
        if self.msg_kind == 'gin':

            if self.id_scope == 'local':
                # dummy variable for central node
                self.central_node_id_encoder = central_encoder(kwargs['id_embedding'], d_id, extend=kwargs['extend_dims'])
                d_id = self.central_node_id_encoder.d_out

            msg_input_dim = None
            update_input_dim = d_in + d_id
            
            self.initial_eps = eps
            if train_eps:
                self.eps = torch.nn.Parameter(torch.Tensor([eps]))
            else:
                self.register_buffer('eps', torch.Tensor([eps]))
            self.eps.data.fill_(self.initial_eps)
            
            self.msg_fn = None
            
        elif self.msg_kind == 'general': # !!!
            msg_input_dim = 2 * d_in + d_id if id_scope == 'local' else 2 * (d_in + d_id)
            
            self.msg_fn = mlp(
                msg_input_dim,
                d_msg,
                d_h,
                seed,
                activation_name,
                bn)
            
            update_input_dim = d_in + d_msg

        else:
            raise NotImplementedError('msg kind {} is not currently supported.'.format(msg_kind))
            
        self.update_fn = mlp(
            update_input_dim,
            d_up,
            d_h,
            seed,
            activation_name,
            bn)

        return

    def forward(self, x, edge_index, **kwargs):
        
        # prepare input features
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        identifiers, degrees = kwargs['identifiers'], kwargs['degrees']
        degrees = degrees.unsqueeze(-1) if degrees.dim() == 1 else degrees
        if self.degree_as_tag:
            x = torch.cat([x, degrees], -1) if self.retain_features else degrees
        
        n_nodes = x.shape[0]
        if self.msg_kind == 'gin':
            if self.id_scope == 'global':
                identifiers_ii = identifiers
            else:
                identifiers_ii, identifiers = self.central_node_id_encoder(identifiers, n_nodes)
            self_msg = torch.cat((x, identifiers_ii), -1)
                
            out = self.update_fn((1 + self.eps) * self_msg + 
                                 self.propagate(edge_index=edge_index, x=x, identifiers=identifiers)) 
                
        elif self.msg_kind == 'general':
            out = self.update_fn(torch.cat((x, self.propagate(edge_index=edge_index, x=x, identifiers=identifiers)), -1))
        
        else:
            raise NotImplementedError("Message kind {} is not currently supported.".format(self.msg_kind))

        return out

    
    def propagate(self, edge_index, x, identifiers):
        
        
        select = 0 if self.flow == 'target_to_source' else 1 
        aggr_dim = 1 - select
        
        edge_index_i, edge_index_j = edge_index[select, :], edge_index[1 - select, :]
        x_i, x_j = x[edge_index_i, :], x[edge_index_j, :]
        
        if self.id_scope == 'local':
            identifiers_ij = identifiers
            identifiers_i, identifiers_j = None, None
        else:
            identifiers_ij = None
            identifiers_i, identifiers_j = identifiers[edge_index_i, :], identifiers[edge_index_j, :]
        
        n_nodes = x.shape[0]
        msgs = self.message(x_i, x_j, identifiers_i, identifiers_j, identifiers_ij)
        msgs = torch.sparse.FloatTensor(edge_index, msgs, torch.Size([n_nodes, n_nodes, msgs.shape[1]]))
        
        if self.aggr == 'add':
            message = torch.sparse.sum(msgs, aggr_dim).to_dense()
            
        elif self.aggr == 'mean':
            degrees = degree(edge_index[select])
            degrees[degrees==0.0] = 1.0
            message = torch.sparse.sum(msgs, aggr_index).to_dense()
            message = message / degrees.unsqueeze(1)
        
        else:
            raise NotImplementedError("Aggregation kind {} is not currently supported.".format(self.aggr))
        
        return message
        

    def message(self, x_i, x_j, identifiers_i, identifiers_j, identifiers_ij):
        
        
        if self.msg_kind == 'gin':
            if self.id_scope == 'global':
                msg_j = torch.cat((x_j, identifiers_j), -1)
            else:
                msg_j = torch.cat((x_j, identifiers_ij), -1)

        elif self.msg_kind == 'general':
            if self.id_scope == 'global':
                msg_j = torch.cat((x_i, x_j, identifiers_i, identifiers_j), -1)
            else:
                msg_j = torch.cat((x_i, x_j, identifiers_ij), -1)
            msg_j = self.msg_fn(msg_j)
            
        else:
            raise NotImplementedError("Message kind {} is not currently supported.".format(self.msg_kind))
            
        return msg_j
    
    def __repr__(self):
        return '{}(msg_fn = {}, update_fn = {})'.format(self.__class__.__name__, self.msg_fn, self.update_fn)



class QGSN_sparse(nn.Module):
    
    def __init__(self,
                 d_in,
                 d_id,
                 d_degree,
                 degree_as_tag,
                 retain_features,
                 id_scope,
                 d_msg,
                 d_up,
                 d_h,
                 seed,
                 activation_name,
                 bn,
                 aggr='add',
                 msg_kind='general',
                 eps = 0,
                 train_eps = False,
                 flow='source_to_target',
                 **kwargs):

        super(QGSN_sparse, self).__init__()
        

        d_msg = d_in if d_msg is None else d_msg

        self.flow = flow
        self.aggr = aggr
        self.msg_kind = msg_kind
        self.id_scope = id_scope
        
        self.degree_as_tag = degree_as_tag
        self.retain_features = retain_features

        if degree_as_tag:
            d_in = d_in + d_degree if retain_features else d_degree

        # different GSN variants: extended gin and general formulation (see supplementary material)
        if self.msg_kind == 'gin':

            if self.id_scope == 'local':
                # dummy variable for central node
                self.central_node_id_encoder = central_encoder(kwargs['id_embedding'], d_id, extend=kwargs['extend_dims'])
                d_id = self.central_node_id_encoder.d_out

            msg_input_dim = None
            update_input_dim = d_in + d_id
            
            self.initial_eps = eps
            if train_eps:
                self.eps = torch.nn.Parameter(torch.Tensor([eps]))
            else:
                self.register_buffer('eps', torch.Tensor([eps]))
            self.eps.data.fill_(self.initial_eps)
            
            self.msg_fn = None
            
        elif self.msg_kind == 'general': # !!!
            msg_input_dim = 2 * d_in + d_id if id_scope == 'local' else 2 * (d_in + d_id)
            
            self.msg_fn = mlp(
                msg_input_dim,
                d_msg,
                d_h,
                seed,
                activation_name,
                bn)
            
            update_input_dim = d_in + d_msg

        else:
            raise NotImplementedError('msg kind {} is not currently supported.'.format(msg_kind))
            
        self.update_fn = mlp(
            update_input_dim,
            d_up,
            d_h,
            seed,
            activation_name,
            bn)

        return

    def forward(self, x, edge_index, **kwargs):
        node_centrality = kwargs['node_centrality']
        edge_centrality = kwargs['edge_centrality']
        
        # prepare input features
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        identifiers, degrees = kwargs['identifiers'], kwargs['degrees']
        degrees = degrees.unsqueeze(-1) if degrees.dim() == 1 else degrees
        if self.degree_as_tag:
            x = torch.cat([x, degrees], -1) if self.retain_features else degrees
        
        n_nodes = x.shape[0]
        if self.msg_kind == 'gin':
            if self.id_scope == 'global':
                identifiers_ii = identifiers
            else:
                identifiers_ii, identifiers = self.central_node_id_encoder(identifiers, n_nodes)
            self_msg = torch.cat((x, identifiers_ii), -1)
                
            out = self.update_fn((1 + self.eps) * self_msg + 
                                 self.propagate(edge_index=edge_index, x=x, identifiers=identifiers)) 
                
        elif self.msg_kind == 'general': # !!!
            out = self.update_fn(torch.cat((x * node_centrality.view(-1, 1), self.propagate(edge_index=edge_index, x=x, identifiers=identifiers, edge_centrality=edge_centrality)), -1))
        
        else:
            raise NotImplementedError("Message kind {} is not currently supported.".format(self.msg_kind))

        return out

    
    def propagate(self, edge_index, x, identifiers, edge_centrality):
        
        
        select = 0 if self.flow == 'target_to_source' else 1 
        aggr_dim = 1 - select
        
        edge_index_i, edge_index_j = edge_index[select, :], edge_index[1 - select, :]
        x_i, x_j = x[edge_index_i, :], x[edge_index_j, :]
        
        if self.id_scope == 'local':
            identifiers_ij = identifiers
            identifiers_i, identifiers_j = None, None
        else:
            identifiers_ij = None
            identifiers_i, identifiers_j = identifiers[edge_index_i, :], identifiers[edge_index_j, :]
        
        n_nodes = x.shape[0]
        msgs = self.message(x_i, x_j, identifiers_i, identifiers_j, identifiers_ij, edge_centrality)
        msgs = torch.sparse.FloatTensor(edge_index, msgs, torch.Size([n_nodes, n_nodes, msgs.shape[1]]))
        
        if self.aggr == 'add':
            message = torch.sparse.sum(msgs, aggr_dim).to_dense()
            
        elif self.aggr == 'mean':
            degrees = degree(edge_index[select])
            degrees[degrees==0.0] = 1.0
            message = torch.sparse.sum(msgs, aggr_index).to_dense()
            message = message / degrees.unsqueeze(1)
        
        else:
            raise NotImplementedError("Aggregation kind {} is not currently supported.".format(self.aggr))
        
        return message
        

    def message(self, x_i, x_j, identifiers_i, identifiers_j, identifiers_ij, edge_centrality):
        
        
        if self.msg_kind == 'gin':
            if self.id_scope == 'global':
                msg_j = torch.cat((x_j, identifiers_j), -1)
            else:
                msg_j = torch.cat((x_j, identifiers_ij), -1)

        elif self.msg_kind == 'general':
            if self.id_scope == 'global':
                msg_j = torch.cat((x_i, x_j, identifiers_i, identifiers_j), -1)
            else:
                msg_j = torch.cat((x_i, x_j, identifiers_ij), -1)
            msg_j = self.msg_fn(msg_j)
            
        else:
            raise NotImplementedError("Message kind {} is not currently supported.".format(self.msg_kind))
        
        msg_j *= edge_centrality.view(-1, 1) # !!!
        return msg_j
    
    def __repr__(self):
        return '{}(msg_fn = {}, update_fn = {})'.format(self.__class__.__name__, self.msg_fn, self.update_fn)


