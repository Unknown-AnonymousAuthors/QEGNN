import argparse

def get_args_parser() -> argparse.ArgumentParser:
    '''
        Get args from the terminal
        Return:
            recieved args from user
    '''
    parser = argparse.ArgumentParser()
    # parser.add_argument("--", type=, default=, required=, help="")

    ''' Upper Setting '''
    parser.add_argument("--model_name", type=str, default='QGCN', help="")
    parser.add_argument("--dataset", type=str, default='MUTAG', help="")

    ''' Model Specific Hyperparams '''

    ''' Training Hyperparams '''
    parser.add_argument("--lr", type=float, default=0.003, help="")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="")
    parser.add_argument("--layer_size", type=int, default=32, help="")
    parser.add_argument("--num_layers", type=int, default=2, help="")
    parser.add_argument("--last_layer", type=str, default='mlp', help="")
    parser.add_argument("--pool", type=str, default='add', help="")
    parser.add_argument("--nc_norm", type=float, default=1.0, help="")
    parser.add_argument("--ec_norm", type=float, default=1.0, help="")
    parser.add_argument("--eig", type=str, default='appro_deg', help="")
    parser.add_argument("--dropout", type=float, default=0.5, help="")
    parser.add_argument("--epochs", type=int, default=1000, help="")
    parser.add_argument("--early_stop", type=int, default=500, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")

    ''' Hardware and Files '''
    parser.add_argument("--gpu", type=str, default="", help="")
    parser.add_argument("--record_path", type=str, default='./records/record.log', help="")
    parser.add_argument("--log_path", type=str, default='./logs', help="")
    parser.add_argument("--animator_output", action='store_true', help="")

    ''' Parser Setting '''
    parser.set_defaults(optimize=False)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    opt = parser.parse_args()
    print(opt)