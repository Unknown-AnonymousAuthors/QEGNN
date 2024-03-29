import torch
import torch_geometric
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import models as models
import os

from tqdm import tqdm
import argparse
import time
import numpy as np
import util

from torch_geometric.utils import dense_to_sparse, to_networkx
from train.csl_classify import train, eval
import subgraph

from torch_geometric.datasets import GNNBenchmarkDataset

def main():
    parser = argparse.ArgumentParser(description="CSL")
    parser.add_argument('--dataset', type=str, default='CSL')
    parser.add_argument('--d', type=int, default=1,
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
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--drop_ratio', type=float, default=0.0,
                        help='dropout ratio')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers')
    parser.add_argument('--emb_dim', type=int, default=100,
                        help='dimensionality of hidden units in GNNs')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='input batch size for training')
    parser.add_argument('--residual', type=bool, default=True,
                        help='residual')
    parser.add_argument('--combination', type=str, default="multi", choices=["sum", "multi"],
                        help='pair combination operation')
    parser.add_argument('--readout', type=str, default="sum", choices=["sum", "mean"],
                        help='readout')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 1000)')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    dataset = GNNBenchmarkDataset(root=f"dataset/{args.dataset}", name=f"{args.dataset}")

    test_split = [[9,10,14,23,26,27,35,39,40,47,50,57,66,70,71,75,82,83,91,93,94,107,110,117,130,132,133,136,140,142],
    [1,4,5,16,21,24,32,33,38,46,48,52,62,72,73,79,80,85,97,101,102,108,109,111,123,127,129,143,147,149],
    [7,11,12,17,20,25,34,43,44,45,54,59,64,65,69,76,77,86,95,99,100,105,116,118,124,126,128,138,139,144],
    [0,2,13,15,19,22,31,36,41,55,56,58,61,67,68,78,81,88,92,96,104,106,113,115,121,125,131,137,145,146],
    [3,6,8,18,28,29,30,37,42,49,51,53,60,63,74,84,87,89,90,98,103,112,114,119,120,122,134,135,141,148],
    ]
    val_split = [[111,79,135,129,80,52,98,53,3,127,16,17,32,102,13,105,106,0,38,64,65,96,19,77,143,147,60,33,56,131],
    [126,136,90,26,98,54,104,77,71,113,106,40,110,144,14,41,146,47,13,86,121,133,88,34,57,27,64,19,68,7],
    [39,88,68,26,29,67,90,0,2,111,113,142,117,122,71,53,41,101,103,49,31,52,140,130,82,5,28,85,121,147],
    [50,71,119,141,23,138,42,64,40,105,75,63,89,53,4,129,38,117,101,147,122,52,17,14,11,86,28,132,93,99],
    [86,121,146,2,12,39,96,19,117,94,71,81,104,65,4,54,118,50,59,69,75,132,140,31,43,22,137,115,126,27],
    ]
    train_split=[[62,69,88,115,43,128,11,55,108,84,137,74,48,104,72,81,24,34,90,12,134,37,99,58,63,25,22,7,4,30,103,21,87,28,148,86,68,44,118,145,36,114,95,54,42,144,18,20,124,8,85,126,122,138,49,73,5,92,141,113,45,109,120,146,51,59,149,89,41,15,46,139,2,116,123,78,61,97,121,112,125,76,101,119,6,1,67,100,31,29],
    [63,65,95,49,55,141,135,105,134,122,61,66,132,89,118,37,99,92,91,142,18,0,93,139,15,116,9,148,3,30,53,43,138,35,84,67,12,51,117,22,125,25,69,74,81,39,120,70,10,17,45,137,124,75,11,87,94,131,140,29,107,58,2,20,82,96,119,83,23,28,112,36,8,130,60,56,115,42,50,59,31,76,100,128,78,44,114,103,145,6],
    [42,137,47,57,10,87,148,83,61,23,107,32,109,73,14,123,46,72,38,37,51,48,24,127,21,92,141,36,132,6,13,136,84,19,98,91,63,120,134,16,106,18,60,50,112,80,125,66,22,135,131,94,4,3,1,79,70,102,110,145,96,33,56,146,108,93,58,143,119,97,15,129,27,8,81,40,149,115,55,75,9,30,78,74,104,89,35,62,133,114],
    [70,90,65,46,44,87,128,114,98,136,33,80,116,149,84,118,7,47,5,103,21,34,57,37,134,25,100,142,3,60,79,97,69,135,8,66,144,45,139,12,148,102,107,35,95,1,108,124,18,82,16,59,143,76,111,10,126,43,77,29,109,9,26,94,73,120,49,6,140,20,83,74,48,110,133,51,72,32,39,130,24,127,91,54,123,85,30,27,62,112],
    [55,26,13,23,46,16,101,136,91,125,111,80,25,9,14,128,56,15,109,100,133,20,40,82,143,21,88,113,38,77,45,139,62,67,130,124,93,10,35,110,73,144,116,44,58,142,107,85,36,83,131,106,64,7,70,138,105,127,108,66,97,79,34,0,147,5,52,61,47,72,92,48,17,1,33,68,24,129,32,78,123,102,95,149,145,11,41,76,57,99],
    ]

    avg_test_acc = []
    avg_train_acc = []
    avg_epochs = []

    t0 = time.time()
    per_epoch_time = []

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for split_number in range(5):
            print("RUN NUMBER: ", split_number)
            trainset, valset, testset = dataset[train_split[split_number]], dataset[val_split[split_number]], dataset[test_split[split_number]]
            print("Training Graphs: ", len(trainset))
            print("Validation Graphs: ", len(valset))
            print("Test Graphs: ", len(testset))
            print("Number of Classes: ", 10)
            
            print('Computing pair infomation...')
            time_t = time.time()
            train_loader = []
            for batch in tqdm(DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate)):
                train_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
            train_loader = DataLoader(train_loader, batch_size=1, shuffle=True)

            valid_loader = []
            for batch in tqdm(DataLoader(valset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate)):
                valid_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
            valid_loader = DataLoader(valid_loader, batch_size=1, shuffle=False)

            test_loader = []
            for batch in tqdm(DataLoader(testset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate)):
                test_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
            test_loader = DataLoader(test_loader, batch_size=1, shuffle=False)
            print('Pair infomation computed! Time:', time.time() - time_t)
                

            params = {
                'nfeat':1,
                'nhid':args.emb_dim, 
                'nclass':10,
                'nlayers':args.num_layer,
                'dropout':args.drop_ratio,
                'readout':args.readout,
                'd':args.d,
                't':args.t, 
                'scalar':args.scalar,
                'mlp':args.mlp, 
                'jk':args.jk, 
                'combination':args.combination,
                'keys':subgraph.get_keys_from_loaders([train_loader, valid_loader, test_loader]),
            }

            model = models.GNN_bench(params).to(device)

            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                factor=0.5,
                                                                patience=5,
                                                                verbose=True)

            if split_number == 0:
                n_params = util.get_n_params(model)
                print('emb_dim:', args.emb_dim)
                print('number of parameters:', n_params)
                assert(n_params <= 500000)

            epoch_train_losses, epoch_val_losses = [], []
            epoch_train_accs, epoch_val_accs = [], [] 
        
            with tqdm(range(args.epochs)) as tq:
                for epoch in tq:

                    tq.set_description('Epoch %d' % epoch)    

                    startime_t = time.time()
                    
                    epoch_train_loss, epoch_train_acc, optimizer = train(model, optimizer, train_loader, device, epoch)
                    epoch_val_loss, epoch_val_acc = eval(model, valid_loader, device, epoch)
                    _, epoch_test_acc = eval(model, test_loader, device, epoch)   
                    
                    epoch_train_losses.append(epoch_train_loss)
                    epoch_val_losses.append(epoch_val_loss)
                    epoch_train_accs.append(epoch_train_acc)
                    epoch_val_accs.append(epoch_val_acc)
                    
                    epoch_train_acc = 100.* epoch_train_acc
                    epoch_test_acc = 100.* epoch_test_acc
                    
                    tq.set_postfix(time=time.time() - startime_t, lr=optimizer.param_groups[0]['lr'],
                                    train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                    train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                    test_acc=epoch_test_acc)  

                    per_epoch_time.append(time.time() - startime_t)

                    scheduler.step(epoch_val_loss)

                    if optimizer.param_groups[0]['lr'] < 1e-5:
                        print("\n!! LR EQUAL TO MIN LR SET.")
                        break

            _, test_acc = eval(model, test_loader, device, epoch)
            _, train_acc = eval(model, train_loader, device, epoch)
            avg_test_acc.append(test_acc)   
            avg_train_acc.append(train_acc)
            avg_epochs.append(epoch)

            print("Test Accuracy [LAST EPOCH]: {:.4f}".format(test_acc))
            print("Train Accuracy [LAST EPOCH]: {:.4f}".format(train_acc))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
        

    print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time()-t0)/3600))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    # Final test accuracy value averaged over 5-fold
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} ± {:.4f}"""          .format(np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100))
    print("\nAll splits Test Accuracies:\n", avg_test_acc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {:.4f} ± {:.4f}"""          .format(np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100))
    print("\nAll splits Train Accuracies:\n", avg_train_acc)

if __name__ == "__main__":
    main()
