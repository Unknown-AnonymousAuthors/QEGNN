import os, sys
import torch
import random
import numpy as np
import warnings
import torch.nn as nn

from models import get_model
from trainer import MyTrainer
from dataset import get_dataset, KFoldIter
from my_utils.utils.dict import Merge, DictIter

sys.path.append(os.getcwd() + '/..')
warnings.filterwarnings("ignore")

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def main(opt):
    args_init = {
        # Training Setting
        'model_name': opt.model_name,
        'dataset': opt.dataset,
        
        # Model Specific Hyperparams 

        # Training Hyperparams 
        'lr': opt.lr,
        'weight_decay': opt.weight_decay,
        'layer_size': opt.layer_size,
        'num_layers': opt.num_layers,
        'last_layer': opt.last_layer,
        'pool': opt.pool,
        'nc_norm': opt.nc_norm,
        'ec_norm': opt.ec_norm,
        'eig': opt.eig,
        'dropout': opt.dropout,
        'epochs': opt.epochs,
        'early_stop': opt.early_stop,
        'batch_size': opt.batch_size,

        # Hardware and Files 
        'gpu': opt.gpu,
        'record_path': opt.record_path,
        'log_path': opt.log_path,
        'animator_output': opt.animator_output,
    }
    record_list = _train(args_init)

def _train(args_init):
    record_list = []
    args_iter = iter(DictIter(args_init))
    for args in args_iter:
        cal_entro = True # if args['model_name'][0] == 'Q' else False
        dataset = get_dataset(args['dataset'], recal=False, cal_entro=cal_entro, nc_norm=args['nc_norm'], ec_norm=args['ec_norm'], eig=args['eig'])
        pre_split = True if args['dataset'] in ['ogbg-molhiv', 'ZINC_full', 'ZINC_subset'] else False
        args['pre_split'] = pre_split
        if not pre_split:
            dataset.shuffle()
        task_type = 'regression' if args['dataset'] in ['ZINC_full', 'ZINC_subset', 'QM9'] else 'classification'
        args['task_type'] = task_type
        
        data_info = dataset.statistic_info()
        args = Merge(args, data_info)
        device = torch.device(f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu")
        dataset.to(device)
        folds_record_list = []
        for train_mask, test_mask in iter(KFoldIter(len(dataset), folds=10)):
            ''' dataset '''
            val_mask = None
            if pre_split:
                train_mask, test_mask, val_mask = torch.load(f"./datasets/obj/{args['dataset']}_masks.pt")
                
            ''' model '''
            model = get_model(args)
            
            ''' train '''
            model.to(device)
            criterion = nn.CrossEntropyLoss(reduction='mean') if task_type == 'classification' else nn.L1Loss(reduction='mean')
            criterion.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) # None
            topic = f"{args['model_name']}_{args['dataset']}_10-folds"
            trainer = MyTrainer(args, model, dataset, criterion, optimizer, topic, device)
            record = trainer(train_mask=train_mask, test_mask=test_mask, val_mask=val_mask, scheduler=scheduler) # List[float] len = 1 or 2
            folds_record_list.append(record)
            
        test_acc = np.array([acc[0] for acc in folds_record_list])
        val_acc = np.array([acc[1] for acc in folds_record_list]) if args['pre_split'] == True else None
        if args['task_type'] == 'classification':
            test_acc *= 100
            val_acc = val_acc * 100 if val_acc is not None else None
            prec_mean, prec_std = 2, 1
        else:
            prec_mean, prec_std = 3, 3
        if val_acc is None:
            print(f"Model {args['model_name']} achieve a test result {test_acc.mean():.{prec_mean}f}±{test_acc.std():.{prec_std}f} on dataset {args['dataset']} with 10-fold setting.")
        else:
            print(f"Model {args['model_name']} achieve a test result {test_acc.mean():.{prec_mean}f}±{test_acc.std():.{prec_std}f} and val result {val_acc.mean():.{prec_mean}f}±{val_acc.std():.{prec_std}f} on dataset {args['dataset']} with pre-split.")
        
    return record_list

if __name__ == "__main__":
    from args import get_args_parser

    parser = get_args_parser()
    opt = parser.parse_args()

    main(opt)