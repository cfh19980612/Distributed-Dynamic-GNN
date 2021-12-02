import torch
import utils as u
import os

class sbm_dataset():
    def __init__(self,args):
        assert args.task in ['link_pred'], 'sbm only implements link_pred'  # assert:断言， 当条件不满足直接返回异常

        # 定义字典，u.Namespace 将字典中对象的调用改为 dict.object
        self.ecols = u.Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                })
        args.sbm_args = u.Namespace(args.sbm_args)

        #build edge data structure
        '''
        注：tensor矩阵切片操作
        edges[:,self.ecols.TimeStep] -> 取出edges矩阵的第四列的所有内容，是一维的tensor矩阵
        '''
        edges = self.load_edges(args.sbm_args, )  # 从文件中按行读取数据，构造成tensor矩阵
        timesteps = u.aggregate_by_time(edges[:,self.ecols.TimeStep], args.sbm_args.aggr_time)  # 首先取edges矩阵的第四列(对应时刻)所有元素
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()

        edges[:,self.ecols.TimeStep] = timesteps
        edges[:,self.ecols.Weight] = self.cluster_negs_and_positives(edges[:,self.ecols.Weight])  # 边赋予权重，大于0的权重赋1，小于0的为0

        # 生成用作训练的数据集，包括类型数，边集，点集，点的特征等
        self.num_classes = edges[:,self.ecols.Weight].unique().size(0)  # edges矩阵中第三列为边的权重，权重有多少不同的数，则有多少类
        self.edges = self.edges_to_sp_dict(edges)  #返回字典， ‘key’：tensor[fromNode，toNode, Timestep] ‘value’：tensor[weight] (0 or 1)
        # print(self.edges['idx'].size(0))
        #random node features
        self.num_nodes = int(self.get_num_nodes(edges))  # 统计点的个数
        self.feats_per_node = args.sbm_args.feats_per_node  # 每个点的特征个数
        self.nodes_feats = torch.rand((self.num_nodes,self.feats_per_node))  # 随机生成点的特征（不随时刻变化？？？）
        self.num_non_existing = self.num_nodes ** 2 - edges.size(0)  # ？？？

    def cluster_negs_and_positives(self,ratings):
        # 张量bool型mask
        pos_indices = ratings >= 0
        neg_indices = ratings < 0

        ratings[pos_indices] = 1
        ratings[neg_indices] = 0
        return ratings

    def prepare_node_feats(self,node_feats):
        node_feats = node_feats[0]
        return node_feats

    def edges_to_sp_dict(self,edges):
        # 取edes的第1，2，4列的全部元素
        idx = edges[:,[self.ecols.FromNodeId,
                       self.ecols.ToNodeId,
                       self.ecols.TimeStep]]

        vals = edges[:,self.ecols.Weight]
        # 返回字典
        return {'idx': idx,
                'vals': vals}

    def get_num_nodes(self,edges):
        all_ids = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        num_nodes = all_ids.max() + 1
        return num_nodes

    def load_edges(self, sbm_args, starting_line = 1):
        file = os.path.join(sbm_args.folder,sbm_args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
            # if  == 0: lines = lines[:2507303]
            # if  == 1: lines = lines[2507302:]
        edges = [[float(r) for r in row.split(',')] for row in lines[starting_line:]]

        # key point: torch.tensor(data) 是一个函数，将data数据根据类型转换为tensor张量，而torch.Tensor是类
        edges = torch.tensor(edges,dtype = torch.long)
        return edges

    def make_contigous_node_ids(self,edges):
        new_edges = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges
