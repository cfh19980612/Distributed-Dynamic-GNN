import utils as u
import torch
import copy
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from collections import OrderedDict
from torch.nn import init

import torch.distributed as dist
from torch.distributed import ReduceOp
# rpc_backend_options = TensorPipeRpcBackendOptions()
# rpc_backend_options.init_method = "tcp://localhost:12349"

'''
注： EGCN用来预测时序输出，即输入一个时序图，预测最后一时刻的图上的节点embedding
->  在构造训练，验证和测试集时，数据集中每一个数据都是某一时刻的图，在传入EGCN之前
->  需要将该图与前N个时刻的图拼接组成一个时序图，用作EGCN的输入，其中N为num_hist_step;
->  例如： 训练集为5~16时刻的graph，则一共有11个训练样本，对于样本1（即5时刻的图），
->  首先构造时序图，即[0~5]时刻的图（参考link_pred_tasker.py中的get_sample（）方法）;
->  之后将该时序图作为EGCN网络输入生成最后一时刻的图embedding，进行预测
'''

PARAMETER_DICT = {}

class EGCN(torch.nn.Module):
    def __init__(self, args, partition, tasker, rank, world_size, remote_module, activation, device='cpu', skipfeats=False):
        super(EGCN,self).__init__()
        GRCU_args = u.Namespace({})  #将字典的索引改为 dict['key'] -> dict.key
        # print('feat_per_node,',tasker.feats_per_node)
        # if feature partition
        self.partition = partition
        feats_per_node = tasker.feats_per_node
        if self.partition == 'feature':
            if rank != world_size - 1:
                feats_per_node = tasker.feats_per_node // world_size
            else:
                # feats_per_node = tasker.feats_per_node // world_size + tasker.feats_per_node%world_size
                feats_per_node = tasker.feats_per_node // world_size
        print(feats_per_node)
        feats = [feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.tasker = tasker
        self.rank = rank
        self.skipfeats = skipfeats
        self.GRCU_layers = nn.ModuleList()
        # self.GCN_init_weights = []
        self.remote_module = remote_module
        self.device = device
        self.partition = partition

        if self.remote_module is not None:
            self.remote_module.cuda(rank)

        self.GCN_list = []
        # 定义EGCN，包含两层，每层的输入特征维度以及输出特征维度，激活函数
        for i in range(1,len(feats)):
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],  # 输入节点特征维度 node_embedding
                                     'out_feats': feats[i],  # 输出节点特征维度
                                     'activation': activation})

            self.GRCU_layers.append(GRCU(GRCU_args, i))
            # self.GCN_init_weights.append(Parameter(torch.Tensor(feats[i-1],feats[i])).cuda(rank))
            # self.reset_param(self.GCN_init_weights[i-1])

    # A_list: 所有时刻（训练样本所在时刻以及前num_hist_step时刻）的图拉普拉斯矩阵； Node_list:上一层所有时刻节点的特征； node_mask_list:
    def forward(self,gcn_init,A_list, Nodes_list, nodes_mask_list):

        node_feats= Nodes_list[-1] # Node_list（hist_ndFeat_list）存储每一时刻节点的输出embedding，-1表示最新（上一）时刻的节点输出embedding

        for (layer, unit) in enumerate(self.GRCU_layers):
            gcn_weights = gcn_init.out_paras(layer)
            # GRCU层会输出该层每个时刻图节点的embedding，该操作会覆盖，使得Nodes_list始终存储最后一层输出
            gcn_weights, Nodes_list = unit(A_list,Nodes_list,nodes_mask_list,self.rank,GCN_init_weights = gcn_weights)

            if layer == 0 and self.partition == 'feature':
                for i in range (len(Nodes_list)):
                    node_embedding = Nodes_list[i]
                    dist.all_reduce(node_embedding, op=ReduceOp.SUM)
                    torch.cuda.synchronize(device=self.rank)
                    # node_embedding.wait()
                    Nodes_list[i] = node_embedding
                    # print(node_embedding)
            # print(self.rank,': reduce complete!')
            self.GCN_list.append(gcn_weights[-1])

            # 在用snapshot partition时，需要将每一层的输出结果都保存，不能覆盖，因为每一层RNN都要通讯
        out = Nodes_list  # 返回最后一时刻的图节点的最后一层embedding输出
        if self.skipfeats:  # ？？？？
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input
        return out

    def GCN_propogation(self):
        return self.GCN_list[-1]

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        # t.data.uniform_(-stdv,stdv)
        init.xavier_uniform_(t)

###  GRCU为EGCN的一层，GRCU输出该层所有时刻的图节点embedding， 在EGCN forward函数中， Nodes_list会由最后一层的GRCU输出覆盖，即最后一层所有时刻图节点的embedding
###  Nodes_list[-1]则为最后一层中，最后一时刻的图节点embedding
class GRCU(torch.nn.Module):
    def __init__(self,args, layer):
        super(GRCU,self).__init__()
        self.args = args
        self.layer = layer
        cell_args = u.Namespace({})       #-------------------------------------------------------------------------------
        cell_args.rows = args.in_feats    #| 对应GCN权重矩阵行和列，行为该层输入节点的embedding维度，列为该层输出节点的embedding维度|
        cell_args.cols = args.out_feats   #-------------------------------------------------------------------------------
        self.evolve_weights = mat_GRU_cell(cell_args,layer)  # 实例化，每个GRU的计算为mat_GRU_cell实例: 利用上一时刻的GCN权重计算该时刻的GCN权重矩阵
        # self.remote_module = remote_module  # 远端模块
        self.activation = self.args.activation  # 激活函数
        # self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))  # 创建GCN的权重参数,时刻0的参数，后续时刻的GCN参数都是算出来的
        # self.GCN_init_weights = gcn(self.args)
        # self.reset_param(self.GCN_init_weights)  # 初始化GCN权重参数
        # PARAMETER_DICT['Layer-{} GCN'.format(layer)] = self.GCN_init_weights

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        # t.data.uniform_(-stdv,stdv)
        init.xavier_uniform_(t)

    # def parameters(self):
    #     return self.GCN_init_weights
    # l+1层前向
    def forward(self,A_list,node_embs_list,mask_list,rank,GCN_init_weights = None, remote_module = None):

        if remote_module is not None:
            # print('remote module')
            GCN_weights = remote_module.forward(self.layer).cuda(rank)
        else:
            # print('local GCN')
            GCN_weights = GCN_init_weights
        # print('Trainer{} layer{}, GCN{}'.format(rank,self.layer,GCN_weights))
        # print(GCN_weights)
        out_seq = []  #该层的输出embedding列表
        weight_seq = []  # 该层每一时刻的GCN参数
        for t,Ahat in enumerate(A_list): # 计算t时刻的隐藏状态（该时刻GCN的参数）以及节点embedding
            node_embs = node_embs_list[t] # 读取t时刻的上一层节点特征H_t^l
            # print(node_embs)
            #first evolve the weights from the initial and use the new weights with the node_embs
            # print('before:',GCN_weights)
            GCN_weights = self.evolve_weights(GCN_weights,node_embs,mask_list[t])  # GRU计算
            # print('after:',GCN_weights)
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))  # GCN计算：A^XW
            # node_embs = self.activation(self.GCN_init_weights(node_embs, Ahat))
            out_seq.append(node_embs)  # 输出该层该时刻的节点输出，加入到该层的输出embedding列表中
            weight_seq.append(GCN_weights)

        return weight_seq, out_seq  # 返回该层【0~t+1】时刻的图节点embedding以及GCN参数

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args,layer):  # 传入的参数为GCN权重参数的行和列，用来定义GRU的权重参数维度，因为GRU的输入为GCN的权重参数
        super(mat_GRU_cell,self).__init__()
        self.args = args
        print('self dimension',self.args.rows, self.args.cols)
        print('self dimension',args.rows, args.cols)
        # 实例化GRCU中GRU操作门:更新门，重置门，候选隐藏状态，以及K-top挑选
        self.update = mat_GRU_gate(args.rows,     # GRU更新门
                                   args.cols,
                                   torch.nn.Sigmoid(),layer,'update')

        self.reset = mat_GRU_gate(args.rows,      # GRU重置门
                                   args.cols,
                                   torch.nn.Sigmoid(),layer,'reset')

        self.htilda = mat_GRU_gate(args.rows,     # GRU候选隐藏状态
                                   args.cols,
                                   torch.nn.Tanh(),layer,'htilda')

        self.choose_topk = TopK(feats = args.rows,  # 选择GCN隐藏层输出的K行，保持隐藏状态与GCN权重的列数相同，k=GCN_weight.cols
                                k = args.cols,layer=layer)

    def forward(self,prev_Q,prev_Z,mask):  # 传入参数， prev_Q：上一时刻(t)GCN的参数； prev_Z：该时刻上一层的node_embedding; mask:该时刻的掩码
        z_topk = self.choose_topk(prev_Z,mask)

        update = self.update(z_topk,prev_Q)  # update = sigmoid(V_z*H_t^l + U_z*W_t + B_z)
        reset = self.reset(z_topk,prev_Q)  # reset = sigmoid(V_r*H_t^l + U_r*W_t + B_r)

        h_cap = reset * prev_Q             # hid^hat = tanh(V_h*H_t^l + U_h*(reset \dot W_t) + B_h)
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap  # hidden = (1 - update) \dot W_t + update \dot hid^hat

        return new_Q


class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation,layer,gate):
        super(mat_GRU_gate,self).__init__()
        self.activation = activation
        #(row x row) x (row x col) = (rwo x col)，即GCN参数矩阵维度
        # self.W = Parameter(torch.Tensor(rows,rows))
        self.W = Parameter(torch.Tensor(rows,rows))
        PARAMETER_DICT['Layer-{} GRU-{}-W'.format(layer,gate)] = self.W
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        PARAMETER_DICT['Layer-{} GRU-{}-U'.format(layer,gate)] = self.U
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))
        PARAMETER_DICT['Layer-{} GRU-{}-B'.format(layer,gate)] = self.bias

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        # t.data.uniform_(-stdv,stdv)
        init.xavier_uniform_(t)

    def forward(self,x,hidden):  # x: 该时刻上一层（l）节点embedding的k列； hidden:上一时刻的GCN权重参数
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k,layer):
        super(TopK,self).__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))  # p
        PARAMETER_DICT['Layer-{} GRU-p'.format(layer)] = self.scorer
        self.reset_param(self.scorer)
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        # t.data.uniform_(-stdv,stdv)
        init.xavier_uniform_(t)

    def forward(self,node_embs,mask):

        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices,self.k)
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

        #we need to transpose the output, 需要转置，因为挑选x矩阵的k行之后，x的维度变为了col(w_GCN的列数) x row(w_GCN的行数，即输入embedding的长度)，转置之后大小与w_GCN完全相同
        return out.t()