import os, sys
import torch
import random
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset
from k_gnn import DataLoader, GraphConv, QGraphConv, avg_pool

from dataset import get_dataset, SepDataset, KFoldIter

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='MUTAG', help="")
parser.add_argument("--quantum", type=str, default='True', help="")
parser.add_argument("--nc_norm", type=float, default=1.0, help="")
parser.add_argument("--ec_norm", type=float, default=10.0, help="")
parser.set_defaults(optimize=False)
parser.set_defaults(kitti_crop=False)
parser.set_defaults(absolute_depth=False)
opt = parser.parse_args()
DATASET = opt.dataset
QUANTUM = opt.quantum
QUANTUM = True if QUANTUM == 'True' else False

# DATASET = 'MUTAG' # ['MUTAG', 'PTC_MR', 'PROTEINS', 'NCI1', 'NCI109', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI']
# QUANTUM = False # True False

BATCH = 128
HIDDEN_SIZE = 64
LEARNING_RATE = 0.01
DROPOUT = 0.5
EPOCHS = 200
EARLY_STOP = 50

NC_NORM = opt.nc_norm # 1.0
EC_NORM = opt.ec_norm # 10.0

print(f"Dataset: {DATASET}, Quantum: {QUANTUM}")

# ENEMYZE with oom one-hot tensor;  PROTEINS with unknown bugs
dataset = get_dataset(dataset_name=DATASET, recal=False, cal_entro=True, nc_norm=NC_NORM, ec_norm=EC_NORM, eig="appro_deg_ge0")
num_i_2, num_i_3 = dataset.iso_onehot()
dataset.shuffle()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, HIDDEN_SIZE)
        self.conv2 = GraphConv(HIDDEN_SIZE, HIDDEN_SIZE)
        self.conv3 = GraphConv(HIDDEN_SIZE, HIDDEN_SIZE)
        self.conv4 = GraphConv(HIDDEN_SIZE + num_i_2, HIDDEN_SIZE)
        self.conv5 = GraphConv(HIDDEN_SIZE, HIDDEN_SIZE)
        self.conv6 = GraphConv(HIDDEN_SIZE + num_i_3, HIDDEN_SIZE)
        self.conv7 = GraphConv(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc1 = torch.nn.Linear(3 * HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = torch.nn.Linear(HIDDEN_SIZE, int(HIDDEN_SIZE / 2))
        self.fc3 = torch.nn.Linear(int(HIDDEN_SIZE / 2), dataset.num_classes)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index))
        data.x = F.elu(self.conv2(data.x, data.edge_index))
        data.x = F.elu(self.conv3(data.x, data.edge_index))
        x = data.x
        x_1 = scatter_add(data.x, data.batch, dim=0)

        data.x = avg_pool(x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)

        data.x = F.elu(self.conv4(data.x, data.edge_index_2))
        data.x = F.elu(self.conv5(data.x, data.edge_index_2))
        x_2 = scatter_mean(data.x, data.batch_2, dim=0)

        data.x = avg_pool(x, data.assignment_index_3)
        data.x = torch.cat([data.x, data.iso_type_3], dim=1)

        data.x = F.elu(self.conv6(data.x, data.edge_index_3))
        data.x = F.elu(self.conv7(data.x, data.edge_index_3))
        x_3 = scatter_mean(data.x, data.batch_3, dim=0)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class QNet(torch.nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.conv1 = QGraphConv(dataset.num_features, HIDDEN_SIZE)
        self.conv2 = QGraphConv(HIDDEN_SIZE, HIDDEN_SIZE)
        self.conv3 = QGraphConv(HIDDEN_SIZE, HIDDEN_SIZE)
        self.conv4 = QGraphConv(HIDDEN_SIZE + num_i_2, HIDDEN_SIZE)
        self.conv5 = QGraphConv(HIDDEN_SIZE, HIDDEN_SIZE)
        self.conv6 = QGraphConv(HIDDEN_SIZE + num_i_3, HIDDEN_SIZE)
        self.conv7 = QGraphConv(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc1 = torch.nn.Linear(3 * HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = torch.nn.Linear(HIDDEN_SIZE, int(HIDDEN_SIZE / 2))
        self.fc3 = torch.nn.Linear(int(HIDDEN_SIZE / 2), dataset.num_classes)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.node_centrality1, data.edge_centrality1))
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.node_centrality1, data.edge_centrality1))
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.node_centrality1, data.edge_centrality1))
        x = data.x
        x_1 = scatter_add(data.x, data.batch, dim=0)

        data.x = avg_pool(x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)

        data.x = F.elu(self.conv4(data.x, data.edge_index_2, None, None))
        data.x = F.elu(self.conv5(data.x, data.edge_index_2, None, None))
        x_2 = scatter_mean(data.x, data.batch_2, dim=0)

        data.x = avg_pool(x, data.assignment_index_3)
        data.x = torch.cat([data.x, data.iso_type_3], dim=1)

        data.x = F.elu(self.conv6(data.x, data.edge_index_3, None, None))
        data.x = F.elu(self.conv7(data.x, data.edge_index_3, None, None))
        x_3 = scatter_mean(data.x, data.batch_3, dim=0)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if QUANTUM:
    model = QNet().to(device)
else:
    model = Net().to(device)

def train(epoch, loader, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def val(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += F.nll_loss(model(data), data.y, reduction='sum').item()
    return loss_all / len(loader.dataset)


def test(loader):
    model.eval()
    correct = 0
    loss_all = 0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        correct += pred.max(1)[1].eq(data.y).sum().item()
        loss_all += F.nll_loss(pred, data.y, reduction='sum').item()
    return correct / len(loader.dataset), loss_all / len(loader.dataset)


acc = []
for i, (train_mask, test_mask) in enumerate(iter(KFoldIter(len(dataset), folds=10))):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=15, min_lr=0.00001)

    train_dataset = dataset[train_mask]
    test_dataset = dataset[test_mask]
    
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH)
    
    print('---------------- Split {} ----------------'.format(i+1))

    best_test_acc = 0.0
    best_test_loss = 9999.0
    early_count = 0
    with tqdm(range(EPOCHS)) as tq:
        for epoch in tq:
            lr = scheduler.optimizer.param_groups[0]['lr']
            train_loss = train(epoch, train_loader, optimizer)
            test_acc, test_loss = test(test_loader)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                early_count = 0
            else:
                early_count += 1
                
            # if epoch % 25 == 0:
            #     print('Epoch: {:03d}, Train Loss: {:.4f}, Test Acc: {:.4f}, Best Acc:{:.4f}'.format(epoch, train_loss, test_acc, best_test_acc))
            tq.set_description(f'Epoch: {epoch}, train_loss={train_loss:.4f}, test_acc={test_acc:.4f}, best_test_acc={best_test_acc:.4f}')  
                
            if early_count >= EARLY_STOP:
                break
            if best_test_acc == 1.0:
                break

    print(f"Fold: {i+1}, Best Acc: {best_test_acc:.4f}, Best Loss: {best_test_loss:.4f}")
    acc.append(best_test_acc)
    
acc = torch.tensor(acc)
print('---------------- Final Result ----------------')
f = open('result.txt', 'a')
result = f'Dataset: {DATASET}, Quantum: {QUANTUM}, Nc_norm: {NC_NORM:.2f}, Ec_norm: {EC_NORM:.2f}, Mean: {acc.mean():7f}, Std: {acc.std():7f}\n'
f.write(result)
print(result)
