import json
from data_loader.forecast_dataloader import ForecastDataset, de_normalized
from models.base_model import HLGCN
from utils.utils import *
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import utils.math_utils 
from utils.math_utils import evaluate
from sklearn.neighbors import kneighbors_graph
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, embd_value,Graph_adj, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_lhgnn.pt')
    value_name = os.path.join(model_dir, epoch + '_embd_value.pt')
    graph_name = os.path.join(model_dir, epoch + '_graph_value.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)
    with open(value_name, 'wb') as f:
        torch.save(embd_value, f)
    with open(graph_name, 'wb') as f:
        torch.save(Graph_adj, f)

def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_lhgnn.pt')
    value_name = os.path.join(model_dir, epoch + '_embd_value.pt')
    graph_name = os.path.join(model_dir, epoch + '_graph_value.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    if not os.path.exists(value_name):
        return
    if not os.path.exists(graph_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    with open(value_name, 'rb') as f:
        embd_value = torch.load(f)
    with open(graph_name, 'rb') as f:
        Graph_adj = torch.load(f)
    return model, embd_value, Graph_adj


def inference(model, position_embed, adj_mx, dataloader, device):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):

            inputs = inputs.to(device)
            target = target.to(device)
            forecast_result, _, _ = model(inputs,adj_mx, position_embed)
            forecast_set.append(forecast_result.detach().cpu().numpy())
            target_set.append(target.detach().cpu().numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)


def validate(model, position_embed, adj_mx, dataloader, device, normalize_method, statistic,
             normalize_column, result_file=None):
    
    forecast_norm, target_norm = inference(model, position_embed, adj_mx, dataloader, device)
    if normalize_method and statistic:
        forecast = de_normalized(forecast_norm, normalize_method, statistic, normalize_column)
        target = de_normalized(target_norm, normalize_method, statistic, normalize_column)
    else:
        forecast, target = forecast_norm, target_norm
    score = evaluate(forecast, target)

    
    score_norm = evaluate(forecast_norm, target_norm)
    print(f'NORM: MAPE {score_norm[0]:7.9%}; MAE {score_norm[1]:7.9f}; RMSE {score_norm[2]:7.9f}.')
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    if result_file:
        if not os.path.exists(result_file):
            os.makedirs(result_file)

        steperror_mae = []
        steperror_mape = []
        steperror_rmse = []
        for i in range(forecast.shape[1]):
            steperror_mape.append(evaluate(forecast[:,i,:],target[:,i,:])[0])
            steperror_mae.append(evaluate(forecast[:,i,:],target[:,i,:])[1])
            steperror_rmse.append(evaluate(forecast[:,i,:],target[:,i,:])[2])
        steperror = np.vstack((np.array(steperror_mape),np.array(steperror_mae),np.array(steperror_rmse)))
        np.savetxt(f'{result_file}/steperror.csv', steperror, delimiter=",")

        step_to_print = int(forecast.shape[1]/2)
        forcasting_2d = forecast[:, step_to_print, :]
        forcasting_2d_target = target[:, step_to_print, :]

        np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
        np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
        np.savetxt(f'{result_file}/predict_abs_error.csv',
                   np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
        np.savetxt(f'{result_file}/predict_ape.csv',
                   np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

    return dict(mae=score[1], mape=score[0], rmse=score[2])


def train(train_data, valid_data, test_data, args, result_file):
    # train_data:numpy [3499,140]
    node_cnt = train_data.shape[1]
    window_size = args.window_size + args.position_emb_dim
    model = HLGCN(node_cnt, args.input_dim, window_size, args.horizon, args.dropout_rate, args.HLfilter_layers,
                    args.glu_out_dim, args.lhgcn_out_dim, args.last_fc_unit, args.use_padding, args.hlgcn_block_layers,
                    args.TimeBlock_kernel,args.position_emb_dim)
    model.to(device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    if args.norm_column_wise:
        if args.norm_method == 'z_score':
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
        elif args.norm_method == 'min_max':
            train_min = np.min(train_data, axis=0)
            train_max = np.max(train_data, axis=0)
            normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
        else:
            normalize_statistic = None
        if normalize_statistic is not None:
            with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
                json.dump(normalize_statistic, f)
    else:
        if args.norm_method == 'z_score':
            train_mean = np.mean(train_data)
            train_std = np.std(train_data)
            normalize_statistic = {"mean": train_mean, "std": train_std}
        elif args.norm_method == 'min_max':
            train_min = np.min(train_data)
            train_max = np.max(train_data)
            normalize_statistic = {"min": train_min, "max": train_max}
        else:
            normalize_statistic = None
        if normalize_statistic is not None:
            with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
                json.dump(normalize_statistic, f)

    if args.norm_method == 'z_score':
        scaler = utils.math_utils.StandardScaler(mean=train_mean, std=train_std)
    else:
        scaler = utils.math_utils.MinMaxScaler(min=train_min, max=train_max)
    train_feas = scaler.transform(train_data)  # (23990, 207)

    k = args.knn
    knn_metric = 'cosine'
    g = kneighbors_graph(train_feas[100:2100,:].T, k, metric=knn_metric)
    g = np.array(g.todense(), dtype=np.float32)
    adj_mx = torch.Tensor(g).to(device)
        
    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    #learning rate decay
    my_lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=my_optim, milestones=lr_decay_steps, gamma=args.lr_decay_rate)

    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic, normalize_column=args.norm_column_wise)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic, normalize_column=args.norm_column_wise)

    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=True, shuffle=True,
                                         num_workers=0)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, drop_last=True, shuffle=True,
                                         num_workers=0)

    test_set = ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method, norm_statistic=normalize_statistic, normalize_column=args.norm_column_wise)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=True, shuffle=False, 
                                        num_workers=0)

    if args.loss_function == 'mae':
        forecast_loss = nn.L1Loss(reduction='mean').to(args.device)  # mae
    elif args.loss_function == 'mask_mae':
        forecast_loss = utils.math_utils.MAE_torch  #h
    elif args.loss_function == 'mse':
        forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
    else:
        forecast_loss = nn.L1Loss(reduction='mean').to(args.device)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Device:{os.environ['CUDA_VISIBLE_DEVICES']}, Total Trainable Params: {total_params}")

    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    performance_metrics = {}
    graph_edge = 0

    train_losses = []
    graph_edges = []
    valid_mae = []
    valid_rmse = []
    valid_mape = []
    test_mae = []
    train_graph = 0
    # position_embd = torch.zeros(args.batch_size, args.lhgcn_out_dim, args.num_nodes, args.position_emb_dim).to(device)    
    position_embd = torch.ones(args.batch_size, args.lhgcn_out_dim, args.num_nodes, args.position_emb_dim).to(device)
    #position_embd = torch.rand((args.num_nodes, args.position_emb_dim)).to(device)

    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        losses = []
        graphedge = []

        if epoch < args.epoch_use_regularization:
            label = 'with_regularization'
        else:
            label = 'without_regularization'

        activeF = nn.ReLU6()
        activeT = nn.Tanh()
        activeR = nn.ReLU()
        mask = torch.eye(args.num_nodes, args.num_nodes).bool().to(device)       

        if args.graph_learning_metric == 0:
            Graph_adj = activeR(activeT(torch.mm(position_embd,position_embd.t())-torch.mm(position_embd.t(),position_embd)))
            Graph_adj = torch.where(Graph_adj <args.rewiring_distance, 1.0, 0.0)
            Graph_adj.masked_fill_(mask, 0)
        elif args.graph_learning_metric == 1:
            Bgraph = torch.mean(position_embd,dim=1)
            Bgraph = torch.mean(Bgraph,dim=0)  
            Graph_adj = torch.norm(Bgraph[:,None]-Bgraph, dim=2,p=2)
            Graph_adj = activeF(Graph_adj)
            Graph_adj = torch.where(Graph_adj < args.rewiring_distance, 1.0, 0.0)
            Graph_adj.masked_fill_(adj_mx.bool(), 1)
            Graph_adj.masked_fill_(mask, 0)

        for i, (inputs, target) in enumerate(train_loader):
        
            inputs = inputs.to(args.device)
            target = target.to(args.device)

            model.zero_grad()
            forecast, graph ,position_embd_learned = model(inputs, Graph_adj, position_embd)

            if label == 'without_regularization':  # or label == 'predictor':
                loss = forecast_loss(forecast, target)
            else:
                loss_1 = forecast_loss(forecast, target)
                pred = torch.sigmoid(Graph_adj.view(Graph_adj.shape[0] * Graph_adj.shape[1])) # Another option: use softmax.
                true_label = adj_mx.view(Graph_adj.shape[0] * Graph_adj.shape[1]).to(device)
                compute_loss = torch.nn.BCELoss()
                loss_g = compute_loss(pred, true_label)            
                loss = loss_1 + args.regular_loss_ratio * loss_g
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
            #print(graph)
            graph_edge = torch.sum(torch.abs(graph)).cpu().detach().numpy()
            losses.append(loss.item())
            graphedge.append(graph_edge)

        if args.graph_save:
            graphshow(graph, epoch, result_file, 0.04)
        if args.model_save:
            print('model saved!!!')
            save_model(model, position_embd_learned, result_file, epoch)  
        
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | graph total edges: {}'.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt, int(np.mean(graphedge))))

        if args.lr_decay:
            my_lr_scheduler.step()

        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = \
                validate(model, position_embd_learned, Graph_adj, valid_loader, args.device, args.norm_method, normalize_statistic,
                         args.norm_column_wise, result_file=result_file
                         )
            if best_validate_mae > performance_metrics['mae']:
                best_validate_mae = performance_metrics['mae']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
                if validate_score_non_decrease_count>args.switch_count and train_graph==0:
                    train_graph = 1
                    print('-----------Stop graph learning!!-----------------')

            if is_best_for_now:
                print('Saved a model !!')
                save_model(model, position_embd_learned, Graph_adj, result_file)

            print('------ validate on data: TEST ------')
            test_performance_metrics = \
                validate(model, position_embd_learned, Graph_adj, test_loader, args.device, args.norm_method, normalize_statistic,
                         args.norm_column_wise, result_file=None)

        if train_graph == 0:
            position_embd = position_embd_learned.detach()
        elif train_graph == 1:
            print('------------------------------Read the best graph!------------------------------------')
            _, best_position_embd, Graph_adj = load_model(result_file)
            position_embd = best_position_embd
            train_graph =2
            print('-----------------------------Start optimizing parameters!-----------------------------')
        else:
            position_embd = best_position_embd
            train_graph =2

        train_losses.append(np.mean(losses)*100)
        graph_edges.append(int(np.mean(graphedge)))
        valid_mae.append(performance_metrics['mae'])
        valid_rmse.append(performance_metrics['rmse'])
        valid_mape.append(performance_metrics['mape'])
        test_mae.append(test_performance_metrics['mae'])
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    metrics = np.vstack((np.array(train_losses),np.array(graph_edges),np.array(valid_mae), np.array(valid_rmse), np.array(valid_mape), np.array(test_mae)))
    np.savetxt(f'{result_file}/metrics.csv', metrics, delimiter=",")
    return performance_metrics, normalize_statistic

def test(test_data, args, result_train_file, result_test_file):
    with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)
    model, position_embed, Graph_adj = load_model(result_train_file)
    node_cnt = test_data.shape[1]
    test_set = ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=True,
                                        shuffle=False, num_workers=0)
    performance_metrics = validate(model, position_embed, Graph_adj, test_loader, args.device, args.norm_method, normalize_statistic,
                       args.norm_column_wise, result_file=result_test_file)
    mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
    print('Performance on test set: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.4f}'.format(mape, mae, rmse))
