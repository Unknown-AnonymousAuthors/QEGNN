import os
from my_utils.utils.dict import DictIter

def generate_shell(gpu_idxs, key, exe_py):
    value = args[key]
    l1 = len(value) if isinstance(value, list) else 1
    l2 = len(gpu_idxs) if isinstance(gpu_idxs, list) else 1
    assert l1 == l2
    if not os.path.exists('shells'):
        os.mkdir('shells')

    for i, value in enumerate(value):
        filename = f'./shells/{i+1}.sh'
        f = open(filename, 'w')
        f.write('date -R\n')
        args_new = args
        args_new[key] = value
        for arg in DictIter(args_new):
            if exe_py == 'main.py':
                arg['gpu'] = f'{gpu_idxs[i]}' if isinstance(gpu_idxs, list) else f':{gpu_idxs}'
                arg['record_path'] = f'./records/record{i+1}.csv'
                arg['log_path'] = f'./logs/logs{i+1}'

            content = f"python {exe_py}"
            for k, v in arg.items():
                if v is False:
                    pass
                elif v is True:
                    content += f' --{k}'
                else:
                    content += f' --{k} {v}'
            content += '\n'
            f.write(content)
        f.write('date -R\n')
        f.close()
        os.system(f'chmod 700 {filename}')

args = {
    'model_name': ['GCN', 'GIN', 'GAT', 'SAGE', 'QGCN', 'QGIN', 'QGAT', 'QSAGE',], 
    # ['GCN', 'QGCN', 'GIN', 'QGIN', 'GAT', 'QGAT', 'SAGE', 'QSAGE',],
    # ['GCN', 'GIN', 'GAT', 'SAGE', 'QGCN', 'QGIN', 'QGAT', 'QSAGE',],
    # ['QGCN', 'QGIN', 'QGAT', 'QSAGE'],
    
    'dataset': ['REDDIT-BINARY', 'REDDIT-MULTI-5K'],
    # ['MUTAG', 'PTC_MR', 'PROTEINS', 'NCI1', 'NCI109', 'ENZYMES', 'ogbg-molhiv', 'IMDB-BINARY', 'IMDB-MULTI', 'QM9', 'ZINC_full', 'ZINC_subset', 'EXP', 'SR', 'CSL']
    # ['MUTAG', 'PTC_MR', 'PROTEINS', 'NCI1', 'NCI109', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI']
    # ['ogbg-molhiv', 'QM9', 'ZINC_full', 'ZINC_subset']
    # ['BZR', 'COX2']
    # ['EXP', 'SR', 'CSL']
    # ['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB', 'DD']
    
    'lr': 0.01,
    'weight_decay': 5e-4,
    'layer_size': 64,
    'num_layers': 5,
    'last_layer': 'lin', # ['lin', 'mlp']
    'pool': 'add', # ['mean', 'add']
    'nc_norm': [1.0], # [-1.0, 0.3, 1.0, 3.0]
    'ec_norm': [10.0], # [-1.0, 1.0, 3.0, 10.0]
    'eig': ['appro_deg_ge0'], # ['np', 'appro_deg', 'appro_deg_ge0', 'betweenness', 'current_flow_betweenness']
    'dropout': 0.5,
    'epochs': 500,
    'early_stop': 50,
    'batch_size': 128,
}

if __name__ == '__main__':
    gpu_idxs = [7, 7]
    key = 'dataset' # model_name dataset
    exe_py = 'main.py' # 'main.py', 'iso.py'
    # 14 [0, 0, 1, 1, 2, 0, 7, 3, 0, 7, 3, 4, 4, 4]
    # 8 [0, 0, 0, 6, 7, 0, 5, 5]

    generate_shell(gpu_idxs, key, exe_py)