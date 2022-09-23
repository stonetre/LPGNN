import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Casual_GLU(nn.Module):
    """
    GLU
    """
    def __init__(self, in_channels, out_channels, timeblock_padding,  Linear_TCN_layers=2,  dropout=0.3, kernel_size=2, start_dilation=1):
        super(Casual_GLU, self).__init__()
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.timeblock_padding = timeblock_padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.TCN_l1 = TCN_L(in_dim=in_channels,out_dim=out_channels,
                            kernel_size=kernel_size,layers=Linear_TCN_layers,timeblock_padding=timeblock_padding, dilation = start_dilation) 
        self.TCN_l2 = TCN_L(in_dim=in_channels,out_dim=out_channels,
                            kernel_size=kernel_size,layers=Linear_TCN_layers,timeblock_padding=timeblock_padding, dilation = start_dilation)
        self.TCN_l3 = TCN_L(in_dim=in_channels,out_dim=out_channels,
                            kernel_size=kernel_size,layers=Linear_TCN_layers,timeblock_padding=timeblock_padding, dilation = start_dilation)

    def forward(self, X):

        res = self.TCN_l3(X)
        out = torch.tanh(self.TCN_l1(X)) * torch.sigmoid(self.TCN_l2(X))
        #out = self.TCN_l1(X) * torch.sigmoid(self.TCN_l2(X))       
        out = F.dropout(out, self.dropout, training=self.training) 
        out = F.relu(out + res)
        #out = F.relu(out)
        # out = F.dropout(self.TCN_l1(X))
        # out = F.relu(out)
        return out


class TCN_L(nn.Module):
    def __init__(self, in_dim=1,out_dim=16, kernel_size=2,layers=2,timeblock_padding=True, dilation = 1):
        super(TCN_L, self).__init__()
        self.layers = layers  # 1
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.timeblock_padding = timeblock_padding
        self.filter_convs = nn.ModuleList()

        new_dilation = dilation
        for i in range(layers):
                # dilated convolutions
                if i == 0:
                    self.filter_convs.append(weight_norm(nn.Conv2d(in_channels=in_dim,
                                                   out_channels=out_dim,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation)))
                else:
                    self.filter_convs.append(weight_norm(nn.Conv2d(in_channels=out_dim,
                                                   out_channels=out_dim,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation)))
                new_dilation =new_dilation* 2
        self.receptive_field = np.power(2,layers)-1
                
    def forward(self, input):

        if self.in_dim == self.out_dim:
            filter = input
        else:
            padnum = self.kernel_size-1
            filter = input
            for i in range(self.layers):
                filter = F.pad(filter,(padnum,0,0,0))
                filter = self.filter_convs[i](filter)  
                padnum = padnum * 2

        if self.timeblock_padding:
            filter = filter
        else:
            filter = filter[:,:,:,self.receptive_field:filter.shape[3]]
        return filter
    

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        x = F.pad(x,(self.chomp_size,0,0,0))
        return x.contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size=(1,kernel_size),
                                           stride=stride, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size=(1,kernel_size),
                                           stride=stride, dilation=dilation))
        self.chomp2 = Chomp1d(padding) 
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
      
        self.downsample = nn.Conv2d(n_inputs, n_outputs, kernel_size=(1,1)) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: [64, 1, 475, 12]
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x) 
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):

        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class HLGraphConv(nn.Module):

    def __init__(self, node_num, in_channels, out_channels, dropout_rate):
        super(HLGraphConv, self).__init__()
        self.node_num = node_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        self.Lweight = nn.Parameter(torch.zeros(size=(self.in_channels, self.out_channels)))
        self.Lweight_by = nn.Parameter(torch.zeros(size=(self.in_channels, self.out_channels)))
        self.Hweight = nn.Parameter(torch.zeros(size=(self.out_channels, self.out_channels)))
        self.Hweight_by = nn.Parameter(torch.zeros(size=(self.out_channels, self.out_channels)))
        self.Lweight_l2 = nn.Parameter(torch.zeros(size=(self.out_channels, self.out_channels)))
        self.Lweight_by2 = nn.Parameter(torch.zeros(size=(self.out_channels, self.out_channels)))

        self.reset_parameters()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.batch_norm = nn.BatchNorm2d(self.node_num)
        self.batch_norm2 = nn.BatchNorm2d(self.node_num)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.Lweight.data)
        nn.init.xavier_normal_(self.Lweight_by.data)
        nn.init.xavier_normal_(self.Hweight.data)
        nn.init.xavier_normal_(self.Hweight_by.data)
        nn.init.xavier_normal_(self.Lweight_l2.data)
        nn.init.xavier_normal_(self.Lweight_by2.data)
    
    def forward(self, input, Lfilter, Hfilter):
        output = torch.einsum("ij,jklm->kilm", [Lfilter, input.permute(2,0,3,1)])
        output = torch.matmul(output, self.Lweight)
        output_bypass = torch.matmul(input.permute(0,2,3,1), self.Lweight_by)
        X = F.relu(self.batch_norm(output + output_bypass))
        output = torch.einsum("ij,jklm->kilm", [Hfilter, X.permute(1,0,2,3)])
        output = torch.matmul(output, self.Hweight)
        output_bypass = torch.matmul(X, self.Hweight_by)   
        X = F.relu(self.batch_norm2(output + output_bypass))
        return X.permute(0,3,1,2)


class HLGCN_Block(nn.Module):
    def __init__(self, node_num, in_channels, out_channels, dropout_rate, parallel_HLGCN_layers):
        super(HLGCN_Block, self).__init__()
        self.out_channels = out_channels
        self.parallel_HLGCN_layers = parallel_HLGCN_layers
        self.HLGraphConv_layers = nn.ModuleList()
        for i in range(parallel_HLGCN_layers):
            self.HLGraphConv_layers.append(HLGraphConv(node_num=node_num, in_channels=in_channels, out_channels=out_channels, 
                                        dropout_rate = dropout_rate))

    def forward(self, input, Lfilter, Hfilter):
        cach =  torch.zeros(input.shape[0], self.out_channels, input.shape[2], input.shape[3]).to(device)
        for i in range(self.parallel_HLGCN_layers):
            out = self.HLGraphConv_layers[i](input, Lfilter, Hfilter)
            cach = cach + out
        out = cach / (self.parallel_HLGCN_layers)
        return(out)

class STGM_Block(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, dropout_rate, TimeBlock_kernel, GLU_HLGCN_blocks_layers,
                 parallel_HLGCN_layers, GLU_Linear_TCN_layers, timeblock_padding, GLU_start_dilation):
        super(STGM_Block, self).__init__()
        self.GLU_HLGCN_blocks_layers = GLU_HLGCN_blocks_layers
        self.Causal_GLU_layer = Casual_GLU(in_channels=in_channels, out_channels=out_channels, timeblock_padding=timeblock_padding, 
                                            Linear_TCN_layers=GLU_Linear_TCN_layers, kernel_size = TimeBlock_kernel, start_dilation = GLU_start_dilation)
        self.HLGCN_block_layer = HLGCN_Block(node_num = num_nodes, in_channels = out_channels, out_channels = spatial_channels, dropout_rate = dropout_rate,
                                            parallel_HLGCN_layers = parallel_HLGCN_layers)
        self.GLU_HLGCN_blocks = nn.ModuleList()
        for i in range(GLU_HLGCN_blocks_layers):
            self.GLU_HLGCN_blocks.append(self.Causal_GLU_layer)
            self.GLU_HLGCN_blocks.append(self.HLGCN_block_layer)

        self.skipConv = TCN_L(in_dim=in_channels,out_dim=spatial_channels,kernel_size=TimeBlock_kernel,
                                layers=GLU_Linear_TCN_layers,timeblock_padding=timeblock_padding)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
    def forward(self, X, Lfilter, Hfilter):
        # X : [32, 1, 307, 24]
        res = self.skipConv(X)
        #print(res.shape)
        # t : [50, 32, 307, 22] 
        out = X
        skip_list = []
        for i in range(self.GLU_HLGCN_blocks_layers):
            out = self.GLU_HLGCN_blocks[i*2](out)
            skip_list.append(out)
            out = self.GLU_HLGCN_blocks[i*2+1](out,Lfilter, Hfilter)
            out = self.dropout(out)
        res = res[:,:,:,res.shape[3]-out.shape[3]:res.shape[3]]
        #out = F.relu(res + out)
        out = self.batch_norm((res + out).permute(0,2,1,3)).permute(0,2,1,3)
        return out, skip_list

def get_laplacian_matrix(graph, normalize=True):
    if normalize:
        D = torch.diag(1/torch.sqrt(torch.sum(graph, dim=1)))
        D = torch.where(torch.isinf(D), torch.zeros(D.shape).to(device), D)
        L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)  
    else:
        D = torch.diag(torch.sum(graph, dim=-1))
        L = D - graph
    A_ = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) + graph
    D_ =  torch.diag(1/torch.sqrt(torch.sum(A_, dim=1)))
    D_ = torch.where(torch.isinf(D_), torch.zeros(D_.shape).to(device), D_)
    A_hat = torch.mm(torch.mm(D_, A_), D_)
    return L, A_hat

class HLfilter_construct(nn.Module):
    def __init__(self):
        super(HLfilter_construct, self).__init__()

    def forward(self, Graph_adj):
        adj = Graph_adj
        L, A_hat = get_laplacian_matrix(adj, normalize=True)
        Lfilter = torch.eye(L.size(0), device=L.device, dtype=L.dtype) - L
        Hfilter = torch.eye(L.size(0), device=L.device, dtype=L.dtype) + L
        return Lfilter, Hfilter, adj

class HLGCN(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output, dropout_rate=0.5, 
                 HLfilter_layers=2,glu_out_dim=32, lhgcn_out_dim=16, last_fc_unit = 2272, 
                 use_padding = True, hlgcn_block_layers=4, TimeBlock_kernel=2, position_emb_dim = 2):
        super(HLGCN, self).__init__()
        self.num_nodes = num_nodes
        self.STGM_blocks = nn.ModuleList()
        self.tcn = nn.ModuleList()
        self.STGM_block_layers = hlgcn_block_layers
        self.GLU_HLGCN_blocks_layers = 1
        self.GLU_Linear_TCN_layers = 1
        self.timeblock_padding = use_padding
        self.position_emb_dim = position_emb_dim
        dilation_exponential = 1
        GLU_start_dilation = 1
        end_fc_in_dim = lhgcn_out_dim * 1
        self.usetime_skip = False
        self.usespace_skip = True

        self.start_conv =  weight_norm(nn.Conv2d(in_channels=num_features, out_channels=lhgcn_out_dim, kernel_size=(1,1)))

        for i in range(self.STGM_block_layers):
            if i==0:
                self.STGM_blocks.append(STGM_Block(in_channels=lhgcn_out_dim, out_channels=glu_out_dim,
                                 spatial_channels=lhgcn_out_dim, num_nodes=num_nodes, 
                                 dropout_rate = dropout_rate,
                                 TimeBlock_kernel = TimeBlock_kernel, GLU_HLGCN_blocks_layers=self.GLU_HLGCN_blocks_layers,
                                 parallel_HLGCN_layers=HLfilter_layers, GLU_Linear_TCN_layers=self.GLU_Linear_TCN_layers,
                                timeblock_padding=self.timeblock_padding, GLU_start_dilation=GLU_start_dilation))                
                self.tcn.append(TemporalConvNet(num_inputs=lhgcn_out_dim, num_channels=[lhgcn_out_dim,lhgcn_out_dim,lhgcn_out_dim,lhgcn_out_dim]))
            else:
                self.STGM_blocks.append(STGM_Block(in_channels=lhgcn_out_dim, out_channels=glu_out_dim,
                                 spatial_channels=lhgcn_out_dim, num_nodes=num_nodes, 
                                 dropout_rate = dropout_rate,
                                 TimeBlock_kernel = TimeBlock_kernel,  GLU_HLGCN_blocks_layers=self.GLU_HLGCN_blocks_layers,
                                 parallel_HLGCN_layers=HLfilter_layers, GLU_Linear_TCN_layers=self.GLU_Linear_TCN_layers,
                                 timeblock_padding=self.timeblock_padding, GLU_start_dilation=GLU_start_dilation))
                self.tcn.append(TemporalConvNet(num_inputs=lhgcn_out_dim, num_channels=[lhgcn_out_dim,lhgcn_out_dim]))
            GLU_start_dilation *= dilation_exponential
  
        if self.usetime_skip and self.usespace_skip:
            last_in_dim = lhgcn_out_dim*(hlgcn_block_layers+1)+glu_out_dim*hlgcn_block_layers*self.GLU_HLGCN_blocks_layers
        if self.usetime_skip and not self.usespace_skip:
            last_in_dim = glu_out_dim*hlgcn_block_layers*self.GLU_HLGCN_blocks_layers+lhgcn_out_dim
        if not self.usetime_skip and self.usespace_skip:
            last_in_dim = lhgcn_out_dim*(hlgcn_block_layers+1)
        if not self.usetime_skip and not self.usespace_skip:
            last_in_dim = lhgcn_out_dim

        self.last_temporal = Casual_GLU(in_channels=last_in_dim, out_channels=end_fc_in_dim, timeblock_padding=self.timeblock_padding, 
                                Linear_TCN_layers=1, kernel_size = TimeBlock_kernel)

        self.HLfilter_construct_module = HLfilter_construct()

        if self.timeblock_padding:
            last_fc_unit = end_fc_in_dim*(num_timesteps_input - position_emb_dim)

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(int(last_fc_unit), int(num_timesteps_output * 2 )),   
            nn.ReLU(),
            nn.Linear(int(num_timesteps_output * 2), num_timesteps_output),
        )

        self.graph_dim_reduction =  weight_norm(nn.Conv2d(in_channels=lhgcn_out_dim, out_channels=1, kernel_size=(1,1)))


    def forward(self, X, Graph_adj, position_embed):
        # X : 32 , 12, 30
        X = X.permute(0, 2, 1) 
        X = torch.unsqueeze(X, dim=1)
        # X : [50, 1, 307, 24]
        Lfilter, Hfilter, adj  = self.HLfilter_construct_module(Graph_adj)
        skip=[]
        stgmskip_cach=[]      
        out1 = self.start_conv(X)  #[32, 32, 210, 30]
        out1 = torch.cat([out1,position_embed],dim=3)

        for i in range(self.STGM_block_layers):
            skip.append(self.tcn[i](out1[:,:,:,:out1.shape[3] - self.position_emb_dim]))
            #skip.append(out1[:,:,:,:out1.shape[3] - self.position_emb_dim])
            out1, stgm_skip = self.STGM_blocks[i](out1, Lfilter, Hfilter)
            for i in range(self.GLU_HLGCN_blocks_layers):
                stgmskip_cach.append(stgm_skip[i][:,:,:,:stgm_skip[i].shape[3] - self.position_emb_dim])

        Bgraph = out1[:,:,:,out1.shape[3] - self.position_emb_dim:out1.shape[3]]
        out1 = out1[:,:,:,:out1.shape[3] - self.position_emb_dim]

        if self.timeblock_padding==False:
            for i in range(self.STGM_block_layers):
                skip[i] = skip[i][:,:,:,skip[i].shape[3]-out1.shape[3]:]
            for i in range(self.STGM_block_layers* self.GLU_HLGCN_blocks_layers):
                stgmskip_cach[i] = stgmskip_cach[i][:,:,:,stgmskip_cach[i].shape[3]-out1.shape[3]:]

        out_skip = []
        if self.usespace_skip:
            out_skip.extend(skip)
        if self.usetime_skip:
            out_skip.extend(stgmskip_cach)
        for i in range(len(out_skip)):
            out1 = torch.cat([out_skip[i], out1], dim = 1) 

        out1 = self.last_temporal(out1).permute(0,2,1,3)
        out1 = out1.reshape((out1.shape[0], out1.shape[1], -1))
        # out3 : [50, 307, 64]
        #print(out1.shape) #4:64 3:128 2:192
        out1 = self.fc(out1)      
        return out1.permute(0, 2, 1), adj, Bgraph


