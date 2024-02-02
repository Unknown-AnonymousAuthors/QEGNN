import random
import os.path as osp
import numpy as np
from torch_geometric.datasets import TUDataset
from dataset import get_dataset
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from train.tu import train, test
import subgraph
import argparse
import os
import models as models
import time
from tqdm import tqdm
import util

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MUTAG') # , choices=["IMDB-BINARY","IMDB-MULTI","NCI1","NCI109","ENZYMES","PROTEINS","PTC_MR","MUTAG"]
    parser.add_argument('--quantum', type=str, default='True', help='')
    parser.add_argument('--nc_norm', type=float, default=1, help='')
    parser.add_argument('--ec_norm', type=float, default=10, help='')
    parser.add_argument('--d', type=int, default=2,
                        help='distance of neighbourhood (default: 1)')
    parser.add_argument('--t', type=int, default=2,
                        help='size of t-subsets (default: 2)')
    parser.add_argument('--scalar', type=bool, default=True,
                        help='learn scalars')
    parser.add_argument('--no-connected', dest='connected', action='store_false',
                        help='also consider disconnected t-subsets')

    parser.add_argument('--mlp', type=bool, default=False,
                        help="mlp (default: False)")
    parser.add_argument('--jk', type=bool, default=True,
                        help="jk")
    parser.add_argument('--drop_ratio', type=float, default=0.5, # 0.1 ?
                        help='dropout ratio')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs')
    parser.add_argument('--combination', type=str, default="multi", choices=["sum", "multi"],
                        help='pair combination operation')
    parser.add_argument('--readout', type=str, default="sum", choices=["sum", "mean"],
                        help='readout')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial lr')
    parser.add_argument('--step', type=int, default=50,
                        help='lr decrease steps')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train')
    parser.add_argument('--device', type=int, default=4, # !!!!!!!!!!!!!!!!!!!!!!!!
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--print_params', type=bool, default=False,
                        help='print number of parameters')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    #set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        print('cuda available with GPU:',torch.cuda.get_device_name(0))

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ds_name = args.dataset
    
    dataset = get_dataset(dataset=ds_name, recal=False, cal_entro=True, nc_norm=args.nc_norm, ec_norm=args.ec_norm, eig="appro_deg_ge0")
    dataset.shuffle()
    
    nfeat = dataset.num_features
    nclass = dataset.num_classes

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)

    val_acc_best_l = []
    for fold, (train_idxs, val_idxs) in enumerate(skf.split(dataset, dataset.data.y)):
        # Split data... following GIN and CIN, only look at validation accuracy
        train_masks = [True] * len(dataset)
        val_masks = [False] * len(dataset)
        for idx in val_idxs:
            train_masks[idx] = False
            val_masks[idx] = True
        train_dataset = dataset[train_masks]
        val_dataset = dataset[val_masks]

        # Prepare batching.
        # print('Computing pair infomation...', end=" ")
        time_t = time.time()
        train_loader = []
        valid_loader = []

        for batch in DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True):
            if dataset.data.x is None:
                batch = dataset.transform(batch)
            train_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
        train_loader = DataLoader(train_loader, batch_size=1, shuffle=True)
        for batch in DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True):
            if dataset.data.x is None:
                batch = dataset.transform(batch)
            valid_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
        valid_loader = DataLoader(valid_loader, batch_size=1, shuffle=False)
        # print('Pair infomation computed! Time:', time.time() - time_t)
        
        quantum = True if args.quantum == 'True' else False
        params = {
                'quantum':quantum,
                'nfeat':nfeat,
                'nhid':args.emb_dim, 
                'nclass':nclass,
                'nlayers':args.num_layer,
                'dropout':args.drop_ratio,
                'readout':args.readout,
                'd':args.d,
                't':args.t, 
                'scalar':args.scalar,  
                'mlp':args.mlp, 
                'jk':args.jk, 
                'combination':args.combination,
                'keys':subgraph.get_keys_from_loaders([train_loader, valid_loader]),
                'tu':True,
            }

        # Setup model.
        model = models.GNN_bench(params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.5)

        if args.print_params:
            print('number of parameters:', util.get_n_params(model))

        val_accs = []
        EARLY_STOP = 50
        early_count = 0
        best_val_loss = 9999.0
        with tqdm(range(args.epochs)) as tq:
            for epoch in tq:
                time_t = time.time()
                train_loss, train_acc = train(train_loader, model, optimizer, device)
                val_loss, val_acc = test(valid_loader, model, device)
                scheduler.step()

                val_accs.append(val_acc)

                tq.set_description(f'train_loss={train_loss:.2f}, train_acc={train_acc:.2f}, val_acc={val_acc:.2f}, time={time.time()-time_t:.2f}')  
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_count = 0
                else:
                    early_count += 1
                
                if early_count >= EARLY_STOP:
                    break
                if val_acc >= 1.0:
                    break
        best_val_acc = np.array(val_accs).max().item()
        val_acc_best_l.append(best_val_acc)
        
        print(f"Fold: {fold}, Best Acc: {best_val_acc:.4f}")
        
    accs = np.array(val_acc_best_l)
    mean_perf, std_perf = accs.mean().item() * 100, accs.std().item() * 100
    result = f"Dataset: {args.dataset}, Quantum: {args.quantum}, nc_norm: {args.nc_norm}, ec_norm: {args.ec_norm}, Mean Acc: {mean_perf:.4f}, Std: {std_perf:.4f}\n"
    print(result)
    f = open('result.txt', 'a')
    f.write(result)


if __name__ == "__main__":
    main()