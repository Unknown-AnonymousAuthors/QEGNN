# QEGNN: Plug-and-play framework for Message Passing GNNs
Multiplying quantum structrual encoding to each message for each GNN layer.
## Examples
### QE GCN
```python
# QE GCN layer (Line 66 in models.py)
def message(...):
    ...
    out = norm_deg.view(-1, 1) * norm_quantum.view(-1, 1) * x_j
    ...
```
### QE GIN
```python
# QE GIN layer (Line 126 and 131 in models.py)
def forward(...):
    ...
    out = out * node_centrality.view(-1, 1)
    ...
def message(...):
    out = x_j * norm_quantum.view(-1, 1)
```
### QE GAT
```python
# QE GAT layer (Line 234 and 239 in models.py)
def edge_update(...):
    ...
    norm_quantum = torch.cat([edge_centrality, node_centrality])
    ...
def message(...):
    ...
    out = norm_quantum.view(-1, 1, 1) * out
    ...
```

### QE SAGE
```python
# QE SAGE layer (Line 310 and 316 in models.py)
def forward(...):
    ...
    x_r = node_centrality.view(-1, 1) * x_r
    ...
def message(...):
    ...
    out = edge_centrality.view(-1, 1) * x_j
    ...
```

### QE G3N
```python
# QE G3N layer (Line 135 and 151 in /G3N-master/layers.py)
def forward(...):
    ...
    h_temp *= edge_centrality
    ...
    h_sum *= node_centrality
    ...
```

### QE GSN
```python
# QE GSN layer (Line 289 and 351 in /GSN-master/graph_filters/GSN_sparse.py)
def forward(...):
    ...
    out = self.update_fn(torch.cat((x * node_centrality.view(-1, 1), self.propagate(edge_index=edge_index, x=x, identifiers=identifiers, edge_centrality=edge_centrality)), -1))
    ...
def message(...):
    ...
    msg_j *= edge_centrality.view(-1, 1)
    ...
```

### QE k-gnn
```python
# QE k-gnn layer (Line 112 and 126 in k-gnn-master/k_gnn/graph_conv.py)
def forward(...):
    ...
    out_col *= edge_centrality.view(-1, 1)
    ...
    out = out + node_centrality.view(-1, 1) * torch.mm(x, self.root_weight)
    ...
```
