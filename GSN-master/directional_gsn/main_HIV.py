"""
    IMPORTING LIBS
"""

import numpy as np
import os
import time
import random
import argparse, json

import torch

import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm


#### MODIFIED CODE HERE
from utils_subgraph_encoding import prepare_subgraph_params
import utils_parsing as parse
#### 

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.HIV_graph_classification.dgn_net import DGNNet
from data.HIV import HIVDataset  # import dataset
from train.train_HIV_graph_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

"""
    GPU Setup
"""


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""


def view_model_param(net_params):
    model = DGNNet(net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('DGN Total parameters:', total_param)
    return total_param


"""
    TRAINING CODE
"""


def train_val_pipeline(dataset, params, net_params):
    t0 = time.time()
    per_epoch_time = []

    DATASET_NAME = dataset.name
    MODEL_NAME = 'DGN'

    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    device = net_params['device']

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = DGNNet(net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_ROCs, epoch_val_ROCs, epoch_test_ROCs = [], [], []

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate, pin_memory=True)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs']), unit='epoch') as t:
            for epoch in t:
                if epoch == -1:
                    model.reset_params()


                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_roc, optimizer = train_epoch(model, optimizer, device, train_loader,
                                                                           epoch)
                epoch_val_loss, epoch_val_roc = evaluate_network(model, device, val_loader, epoch)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_ROCs.append(epoch_train_roc.item())
                epoch_val_ROCs.append(epoch_val_roc.item())


                _, epoch_test_roc = evaluate_network(model, device, test_loader, epoch)

                epoch_test_ROCs.append(epoch_test_roc.item())

                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_ROC=epoch_train_roc.item(), val_ROC=epoch_val_roc.item(),
                              test_ROC=epoch_test_roc.item(), refresh=False)

                per_epoch_time.append(time.time() - start)

                scheduler.step(-epoch_val_roc.item())

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - t0 > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break

                print('')

                #for _ in range(5):
                    #print('Sampled value is ', model.layers[1].towers[0].eigfiltbis(torch.FloatTensor([random.random() for i in range(4)]).to('cuda')))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')


    best_val_epoch = np.argmax(np.array(epoch_val_ROCs))
    best_train_epoch = np.argmax(np.array(epoch_train_ROCs))
    best_val_roc = epoch_val_ROCs[best_val_epoch]
    best_val_test_roc = epoch_test_ROCs[best_val_epoch]
    best_val_train_roc = epoch_train_ROCs[best_val_epoch]
    best_train_roc = epoch_train_ROCs[best_train_epoch]

    print("Best Train ROC: {:.4f}".format(best_train_roc))
    print("Best Val ROC: {:.4f}".format(best_val_roc))
    print("Test ROC of Best Val: {:.4f}".format(best_val_test_roc))
    print("Train ROC of Best Val: {:.4f}".format(best_val_train_roc))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))



def main():
    """
        USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--graph_norm', help="Please give a value for graph_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--expid', help='Experiment id.')
    parser.add_argument('--type_net', default='simple', help='Type of net')
    parser.add_argument('--lap_norm', default='none', help='Laplacian normalisation')

    # dgn params
    parser.add_argument('--aggregators', type=str, help='Aggregators to use.')
    parser.add_argument('--scalers', type=str, help='Scalers to use.')
    parser.add_argument('--towers', type=int, default=5, help='Towers to use.')
    parser.add_argument('--divide_input_first', type=bool, help='Whether to divide the input in first layer.')
    parser.add_argument('--divide_input_last', type=bool, help='Whether to divide the input in last layers.')
    parser.add_argument('--edge_dim', type=int, help='Size of edge embeddings.')
    parser.add_argument('--pretrans_layers', type=int, help='pretrans_layers.')
    parser.add_argument('--posttrans_layers', type=int, help='posttrans_layers.')
    parser.add_argument('--pos_enc_dim', default=0, type=int, help='Positional encoding dimension')
    
    
    #### MODIFED CODE HERE
    parser.add_argument('--root_folder', type=str, default='./')
    parser.add_argument('--directions', type=str, help='Different types of vector fields.')
    parser.add_argument('--id_type', type=str, default='cycle_graph')
    parser.add_argument('--induced', type=parse.str2bool, default=True)
    parser.add_argument('--edge_automorphism', type=str, default='induced')
    parser.add_argument('--k', type=parse.str2list2int, default=[6])
    parser.add_argument('--id_scope', type=str, default='local')
    parser.add_argument('--custom_edge_list', type=parse.str2ListOfListsOfLists2int, default=None)
    parser.add_argument('--directed', type=parse.str2bool, default=False)
    parser.add_argument('--directed_orbits', type=parse.str2bool, default=False)
    parser.add_argument('--multiprocessing', type=parse.str2bool, default=False)
    parser.add_argument('--num_processes', type=int, default=64)
    parser.add_argument('--id_encoding', type=str, default=None)
    #### MODIFED CODE HERE

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # dataset, out_dir
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
        
        
    #### MODIFIED CODE HERE
    args.id_encoding = None if args.id_encoding=='None' else args.id_encoding
    directions = [direction for direction in args.directions.split()]
    subgraph_params = {'edge_automorphism': args.edge_automorphism,
                      'id_scope': args.id_scope,
                      'id_type': args.id_type,
                      'k': args.k,
                      'induced': args.induced,
                      'directed': args.directed,
                      'directed_orbits': args.directed_orbits,
                      'multiprocessing': args.multiprocessing,
                      'num_processes': args.num_processes,
                      'id_encoding': args.id_encoding}
    subgraph_params = prepare_subgraph_params(subgraph_params)
    path = os.path.join(args.root_folder, 'dataset', DATASET_NAME)
    dataset = HIVDataset(DATASET_NAME,
                         pos_enc_dim=int(args.pos_enc_dim),
                         norm=args.lap_norm,
                         path=path,
                         directions=directions,
                         verbose=True,
                         **subgraph_params)
    #### MODIFED CODE HERE
    
    
    
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)

    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.graph_norm is not None:
        net_params['graph_norm'] = True if args.graph_norm == 'True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.aggregators is not None:
        net_params['aggregators'] = args.aggregators
    if args.scalers is not None:
        net_params['scalers'] = args.scalers
    if args.towers is not None:
        net_params['towers'] = args.towers
    if args.divide_input_first is not None:
        net_params['divide_input_first'] = args.divide_input_first
    if args.divide_input_last is not None:
        net_params['divide_input_last'] = args.divide_input_last
    if args.edge_dim is not None:
        net_params['edge_dim'] = args.edge_dim
    if args.pretrans_layers is not None:
        net_params['pretrans_layers'] = args.pretrans_layers
    if args.posttrans_layers is not None:
        net_params['posttrans_layers'] = args.posttrans_layers
    if args.type_net is not None:
        net_params['type_net'] = args.type_net
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = args.pos_enc_dim
        
    D = torch.cat([torch.sparse.sum(g.adjacency_matrix(transpose=True), dim=-1).to_dense() for g in
                       dataset.train.graph_lists])
    net_params['avg_d'] = dict(lin=torch.mean(D),
                                   exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                                   log=torch.mean(torch.log(D + 1)))

    net_params['total_param'] = view_model_param(net_params)
    train_val_pipeline(dataset, params, net_params)


main()
