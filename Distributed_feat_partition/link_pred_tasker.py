import torch
import taskers_utils as tu
import utils as u

import functools
import dill

SCALE = 0.1

class Link_Pred_Tasker():
	'''
	Creates a tasker object which computes the required inputs for training on a link prediction
	task. It receives a dataset object which should have two attributes: nodes_feats and edges, this
	makes the tasker independent of the dataset being used (as long as mentioned attributes have the same
	structure).

	Based on the dataset it implements the get_sample function required by edge_cls_trainer.
	This is a dictionary with:
		- time_step: the time_step of the prediction
		- hist_adj_list: the input adjacency matrices until t, each element of the list 
						 is a sparse tensor with the current edges. For link_pred they're
						 unweighted
		- nodes_feats_list: the input nodes for the GCN models, each element of the list is a tensor
						  two dimmensions: node_idx and node_feats
		- label_adj: a sparse representation of the target edges. A dict with two keys: idx: M by 2 
					 matrix with the indices of the nodes conforming each edge, vals: 1 if the node exists
					 , 0 if it doesn't

	There's a test difference in the behavior, on test (or development), the number of sampled non existing 
	edges should be higher.
	'''
	def __init__(self,args,dataset,world_size, rank):
		self.data = dataset
		#max_time for link pred should be one before
		self.max_time = dataset.max_time - 1
		self.args = args
		self.num_classes = 2
		self.world_size = world_size
		self.rankID = rank

		if args.use_2_hot_node_feats:
			max_deg_out, max_deg_in = tu.get_max_degs(self.args, self.data)  # 统计所有时刻最大的入度和出度
			self.feats_per_node = max_deg_out + max_deg_in
		elif args.use_1_hot_node_feats: # 针对无向图，点的特征长度为最大的度
			max_deg,_ = tu.get_max_degs(self.args, self.data)
			self.feats_per_node = max_deg

		# if feature partition
		if self.args.partition == 'feature':
			if self.rankID != self.world_size - 1:
				self.feats_per_node = self.feats_per_node // self.world_size
			else:
				self.feats_per_node = self.feats_per_node // self.world_size + self.feats_per_node%self.world_size

		# self.get_node_feats = self.build_get_node_feats(args,dataset)
		# self.prepare_node_feats = self.build_prepare_node_feats(args,dataset)
		self.is_static = False

	# 创建prepare_node_feats函数，由于多进程使用嵌套函数有bug，故做修改
	# def build_prepare_node_feats(self,args,dataset):
	# 	if args.use_2_hot_node_feats or args.use_1_hot_node_feats:
	# 		def prepare_node_feats(node_feats):
	# 			return u.sparse_prepare_tensor(node_feats,
	# 										   torch_size= [dataset.num_nodes,
	# 										   				self.feats_per_node])
	# 	else:
	# 		prepare_node_feats = self.data.prepare_node_feats

	# 	return dill.dumps(prepare_node_feats)
	def prepare_node_feats(self, node_feats):
		print(self.data.num_nodes, len(node_feats))
		if self.args.use_2_hot_node_feats or self.args.use_1_hot_node_feats:
			return u.sparse_prepare_tensor(node_feats,
											torch_size= [self.data.num_nodes,
											   			self.feats_per_node])
		else:
			return self.data.prepare_node_feats

	def get_node_feats(self,adj):
		# 在边预测中，用边的连接hot编码定义点的特征
		if self.args.use_2_hot_node_feats:  # 针对有向图

			return tu.get_2_hot_deg_feats(adj,
											max_deg_out,
											max_deg_in,
											self.data.num_nodes)
		elif self.args.use_1_hot_node_feats:  # 针对无向图
			max_deg,_ = tu.get_max_degs(self.args, self.data)
			self.feats_per_node = max_deg
			# @functools.wraps(data)
			if self.args.partition == 'feature':
				feature = tu.get_1_hot_deg_feats(adj,
												max_deg,
												self.data.num_nodes)
				if self.rankID != self.world_size - 1:
					feature['idx'] = feature['idx'][self.feats_per_node*self.rankID:self.feats_per_node*self.rankID -1]
				else:
					feature['idx'] = feature['idx'][self.feats_per_node*self.rankID:]
				return feature
			else:
				return tu.get_1_hot_deg_feats(adj,
												max_deg,
												self.data.num_nodes)
		else:
			# @functools.wraps(data)
			return self.data.nodes_feats


	# def build_get_node_feats(self,args,dataset):
	# 	if args.use_2_hot_node_feats:
	# 		max_deg_out, max_deg_in = tu.get_max_degs(args,dataset)
	# 		self.feats_per_node = max_deg_out + max_deg_in
	# 		def get_node_feats(adj):
	# 			return tu.get_2_hot_deg_feats(adj,
	# 										  max_deg_out,
	# 										  max_deg_in,
	# 										  dataset.num_nodes)
	# 	elif args.use_1_hot_node_feats:
	# 		max_deg,_ = tu.get_max_degs(args,dataset)
	# 		self.feats_per_node = max_deg
	# 		# @functools.wraps(dataset)
	# 		def get_node_feats(adj):
	# 			return tu.get_1_hot_deg_feats(adj,
	# 										  max_deg,
	# 										  dataset.num_nodes)
	# 	else:
	# 		# @functools.wraps(dataset)
	# 		def get_node_feats(adj):
	# 			return dataset.nodes_feats

	# 	return get_node_feats


	def get_sample(self, idx, num_hist_Steps, test, **kwargs):
		hist_adj_list = []  # 历史邻接矩阵列表，每一个元素为一个时刻下的邻接矩阵
		hist_ndFeats_list = []  # 历史节点特征列表，每一个元素为一个NxF的矩阵
		hist_mask_list = []  # 历史掩码列表
		existing_nodes = []  #？？？
		label_sp_list = []
		# print(idx - num_hist_Steps, idx+1)
		for i in range(idx - num_hist_Steps + 1, idx+1):
			# 生成字典，'idx'为边，'vals'为权重
			cur_adj = tu.get_sp_adj(edges = self.data.edges,
								   time = i,
								   weighted = True,
								   time_window = self.args.adj_mat_time_window)
			node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)  # 生成点掩码，边中出现过的点为0，其余为-inf
			node_feats = self.get_node_feats(cur_adj)
			# print(node_feats)
			# # if feature partition
			# if self.args.partition == 'feature':
			# 	scale = self.feats_per_node // self.world_size
			# 	if self.rankID != self.world_size -1:
			# 		node_feats = node_feats[:,self.rankID*scale:(self.rankID+1)*scale]
			# 	else:
			# 		node_feats = node_feats[:,self.rankID*scale:]
				# self.feats_per_node = len(node_feats)

			cur_adj = tu.normalize_adj(adj = cur_adj, num_nodes = self.data.num_nodes)

			hist_adj_list.append(cur_adj)
			hist_ndFeats_list.append(node_feats)
			hist_mask_list.append(node_mask)

		# This would be if we were training on all the edges in the time_window
		# label_adj = tu.get_sp_adj(edges = self.data.edges,
		# 						  time = idx+1,    # 获取idx+1时刻图的label
		# 						  weighted = False,
		# 						  time_window =  self.args.adj_mat_time_window)

			# get labels
			label_adj = tu.get_edge_labels(edges=self.data.edges,
											time = i)
			if test:
				neg_mult = self.args.negative_mult_test
			else:
				neg_mult = self.args.negative_mult_training
			if self.args.smart_neg_sampling:
				existing_nodes = cur_adj['idx'].unique()  # 所有边中出现的点
			else:
				existing_nodes = None
			# if self.args.smart_neg_sampling:
			# 	existing_nodes = torch.cat(existing_nodes)

			# 构造负样本,即不存在边的点对
			# if 'all_edges' in kwargs.keys() and kwargs['all_edges'] == True:
			# 	non_exisiting_adj = tu.get_all_non_existing_edges(adj = label_adj, tot_nodes = self.data.num_nodes)
			# else:
			# 	non_exisiting_adj = tu.get_non_existing_edges(adj = label_adj,
			# 											  number = 5000,
			# 											  tot_nodes = self.data.num_nodes,
			# 											  smart_sampling = self.args.smart_neg_sampling,
			# 											  existing_nodes = existing_nodes)

			# select \theta fraction of edges with labels for training and test
			num = int(SCALE*label_adj['vals'].size(0))

			non_exisiting_adj = tu.get_non_existing_edges(adj = label_adj,
														number = num,
														tot_nodes = self.data.num_nodes,
														smart_sampling = self.args.smart_neg_sampling,
														existing_nodes = existing_nodes)
			# label_adj = tu.get_sp_adj_only_new(edges = self.data.edges,
			# 								   weighted = False,
			# 								   time = idx)

			# select a fraction of edges as the training set or test set; [sc'2021]

			# num = 10000
			label_adj['idx'] = label_adj['idx'][:num,:]
			label_adj['vals'] = label_adj['vals'][:num]
			non_exisiting_adj['idx'] = non_exisiting_adj['idx'][:num,:]
			non_exisiting_adj['vals'] = non_exisiting_adj['vals'][:num]

			label_adj_all = {}
			label_adj_all['idx'] = torch.cat([label_adj['idx'],non_exisiting_adj['idx']])
			label_adj_all['vals'] = torch.cat([label_adj['vals'],non_exisiting_adj['vals']])
			label_sp_list.append(label_adj_all)
		return {'idx': idx,
				'hist_adj_list': hist_adj_list,
				'hist_ndFeats_list': hist_ndFeats_list,
				'label_sp_pos': label_adj,
				'label_sp_neg': non_exisiting_adj,
				'label_sp': label_sp_list,
				'node_mask_list': hist_mask_list}

