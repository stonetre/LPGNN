import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datetime import datetime
import configparser
from models.handler import train, test
import argparse
import pandas as pd
import numpy as np

DATASET = 'PEMS08'
HORIZON = '12'
MODEL = 'LPGNN'
DEVICE = 'cuda'

#get configuration
config_file = 'config/{}_{}_{}.conf'.format(DATASET, HORIZON, MODEL)
print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default=DEVICE)
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--dataset', type=str, default=DATASET)
# data
parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
parser.add_argument('--window_size', type=int, default=config['data']['window_size'])
parser.add_argument('--horizon', type=int, default=config['data']['horizon'])               
parser.add_argument('--train_length', type=float, default=config['data']['train_ratio'])    
parser.add_argument('--valid_length', type=float, default=config['data']['val_ratio'])      
parser.add_argument('--test_length', type=float, default=config['data']['test_ratio'])      
parser.add_argument('--norm_method', type=str, default=config['data']['normalizer'])        
parser.add_argument('--norm_column_wise', type=eval, default=config['data']['column_wise']) 
parser.add_argument('--position_emb_dim', type=int, default=config['data']['position_emb_dim']) 
parser.add_argument('--graph_learning_metric', type=int, default=config['data']['graph_learning_metric'])  # 0=cos 1 = eular

# train
parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
parser.add_argument('--loss_function', type=str, default=config['train']['loss_func'])  # mae mask_mae
parser.add_argument('--epoch', type=int, default=config['train']['epochs'])
parser.add_argument('--switch_count', type=int, default=config['train']['switch_count'])
parser.add_argument('--lr', type=float, default=config['train']['lr_init'])
parser.add_argument('--seed', type=float, default=config['train']['seed'])
parser.add_argument('--lr_decay', type=eval, default=config['train']['lr_decay'])
parser.add_argument('--lr_decay_rate', type=float, default=config['train']['lr_decay_rate'])
parser.add_argument('--lr_decay_step', type=str, default=config['train']['lr_decay_step'])
parser.add_argument('--early_stop', type=eval, default=config['train']['early_stop'])
parser.add_argument('--early_stop_patience', type=eval, default=config['train']['early_stop_patience'])
parser.add_argument('--grad_norm', type=eval, default=config['train']['grad_norm'])
parser.add_argument('--max_grad_norm', type=eval, default=config['train']['max_grad_norm'])
parser.add_argument('--real_value', type=eval, default=config['train']['real_value'])
parser.add_argument('--validate_freq', type=int, default=config['train']['validate_freq'])
parser.add_argument('--optimizer', type=str, default=config['train']['optimizer'])                         # RMSProp, Adam
parser.add_argument('--epoch_use_regularization', type=float, default=config['train']['epoch_regular'])
parser.add_argument('--knn', type=int, default=config['train']['knn_graph_num'])
parser.add_argument('--regular_loss_ratio', type=float, default=config['train']['regular_loss_ratio'])
parser.add_argument('--rewiring_distance', type=float, default=config['train']['rewiring_distance'])
parser.add_argument('--graph_save', type=eval, default=config['train']['graph_save'])
parser.add_argument('--model_save', type=eval, default=config['train']['model_save'])

# model
parser.add_argument('--input_dim', type=int, default=config['model']['input_dim'])
parser.add_argument('--output_dim', type=int, default=config['model']['output_dim'])
parser.add_argument('--glu_out_dim', type=int, default=config['model']['glu_out_dim'])
parser.add_argument('--lhgcn_out_dim', type=int, default=config['model']['lhgcn_out_dim'])
parser.add_argument('--dropout_rate', type=float, default=config['model']['dropout_rate'])            
parser.add_argument('--HLfilter_layers', type=int, default=config['model']['HLfilter_layers']) 
parser.add_argument('--last_fc_unit', type=int, default=config['model']['last_fc_unit']) 
parser.add_argument('--use_padding', type=eval, default=config['model']['use_padding']) 
parser.add_argument('--hlgcn_block_layers', type=int, default=config['model']['hlgcn_block_layers']) 
parser.add_argument('--TimeBlock_kernel', type=int, default=config['model']['TimeBlock_kernel']) 

args = parser.parse_args()
print(f'Training configs: {args}')
data_file = os.path.join('dataset', args.dataset + '.csv')
print(data_file)
result_train_file = os.path.join('output', args.dataset, 'train')
result_test_file = os.path.join('output', args.dataset, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)
data = pd.read_csv(data_file).values

train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
test_ratio = 1 - train_ratio - valid_ratio
train_data = data[:int(train_ratio * len(data))]
valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
test_data = data[int((train_ratio + valid_ratio) * len(data)):]

torch.manual_seed(0)

graph_adj = None
if __name__ == '__main__':

    if args.train:
        try:
            before_train = datetime.now().timestamp()
            _, normalize_statistic = train(train_data, valid_data, test_data, args, result_train_file)
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        test(test_data, args, result_train_file, result_test_file)
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
    print('done')
