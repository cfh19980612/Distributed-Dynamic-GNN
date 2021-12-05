import torch
import utils as u
import time
import pandas as pd
import numpy as np
import os
import copy

from log import logger
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import torch.nn.functional as F

class Trainer():
	def __init__(self,args, splitter, gcn, classifier, comp_loss, dataset, num_classes, device, DIST_DEFAULT_WORLD_SIZE, DIST_DEFAULT_INIT_METHOD, rank):
		self.args = args
		self.splitter = splitter  # 数据集类，包含训练集，验证集和测试集
		self.tasker = splitter.tasker  # 为训练集中的样本生成动态图属性列表，包含时序邻接矩阵列表，时序点特征列表等，生成的矩阵都为稀疏矩阵（字典），作为输入之前需要转为稠密矩阵
		self.gcn = gcn  # 模型
		self.classifier = classifier  # 分类器
		# self.gcn_fp = gcn_fp  # 拆分的GCN
		self.comp_loss = comp_loss  # loss函数

		self.num_nodes = dataset.num_nodes  # 总点数（不随时刻变化）
		self.data = dataset  # 完整数据集（无调用）
		self.num_classes = num_classes  # 总类别数

		# self.logger = logger.Logger(args, self.num_classes)

		self.init_optimizers(args)  #初始化优化器

		self.valid_measure = u.Measure(num_classes=num_classes, target_class=1)
		self.test_measure = u.Measure(num_classes=num_classes, target_class=1)

		self.device = device
		self.DIST_DEFAULT_WORLD_SIZE = DIST_DEFAULT_WORLD_SIZE
		self.DIST_DEFAULT_INIT_METHOD = DIST_DEFAULT_INIT_METHOD
		self.rank = rank
		if self.tasker.is_static:
			adj_matrix = u.sparse_prepare_tensor(self.tasker.adj_matrix, torch_size = [self.num_nodes], ignore_batch_dim = False)
			self.hist_adj_list = [adj_matrix]
			self.hist_ndFeats_list = [self.tasker.nodes_feats.float()]
		if self.args.partition == 'feature':
			if self.rank != self.DIST_DEFAULT_WORLD_SIZE - 1:
				self.feature_per_node = self.tasker.feats_per_node//self.DIST_DEFAULT_WORLD_SIZE
			else:
				self.feature_per_node = self.tasker.feats_per_node // self.DIST_DEFAULT_WORLD_SIZE + self.tasker.feats_per_node%self.DIST_DEFAULT_WORLD_SIZE
	def init_optimizers(self,args):
		# gcn网络优化器，即example中的EGCN网络
		params = self.gcn.parameters()
		self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
		self.gcn_opt.zero_grad()

		# 分类器网络优化器
		params = self.classifier.parameters()
		self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
		self.classifier_opt.zero_grad()

		# if self.args.partition == 'feature':  # feature partition
		# 	params = self.gcn_fp.parameters()
		# 	self.gcn_fp_opt = torch.optim.Adam(params, lr = args.learning_rate)
		# 	self.gcn_fp_opt.zero_grad()

	def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
		torch.save(state, filename)

	def load_checkpoint(self, filename, model):
		if os.path.isfile(filename):
			print("=> loading checkpoint '{}'".format(filename))
			checkpoint = torch.load(filename)
			epoch = checkpoint['epoch']
			self.gcn.load_state_dict(checkpoint['gcn_dict'])
			self.classifier.load_state_dict(checkpoint['classifier_dict'])
			self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
			self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
			# self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
			return epoch
		else:
			# self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
			return 0

	def train(self):
		self.tr_step = 0
		best_eval_valid = 0
		eval_valid = 0
		epochs_without_impr = 0
		log_file = self.args.data

		# to csv
		time_spend = []
		Loss = []
		Precision = []
		Recall = []
		F1 = []
		time_now = time.time()

		# generate the training dataload
		train_data_load = []
		time_data_start = time.time()
		print('[{},{}] | Training data load start!'.format(os.getpid(), self.rank))
		for s in self.splitter.train:
			train_data_load.append(s)
		time_data_end = time.time()
		# print(len(train_data_load))
		print('[{},{}] | Training data load end with cost: {}'.format(os.getpid(), self.rank, time_data_end - time_data_start))

		# generate the test dataload
		test_data_load = []
		time_data_start = time.time()
		print('[{},{}] | Test data load start!'.format(os.getpid(), self.rank))
		for s in self.splitter.test:
			test_data_load.append(s)
		time_data_end = time.time()
		print('[{},{}] | Test data load end with cost: {}'.format(os.getpid(), self.rank, time_data_end - time_data_start))

		for e in range(self.args.num_epochs):
			train_data = iter(copy.deepcopy(train_data_load))  # 使用深拷贝防止数据集处理时被修改
			train_epoch_time_start = time.time()
			loss, nodes_embs = self.run_epoch(train_data, e, 'TRAIN', grad = True)  # 训练一个epoch，参数(训练集，epochID，‘Train’，梯度求解)
			train_epoch_time_end = time.time()

			# 存储时刻以及训练loss
			time_spend.append(train_epoch_time_end-time_now)  # 时刻
			Loss.append(loss)

			# #  是否执行验证集
			# if len(self.splitter.dev)>0 and e>self.args.eval_after_epochs:
			# 	eval_valid, _ = self.run_epoch(self.splitter.dev, e, 'VALID', grad = False)
			# 	if eval_valid>best_eval_valid:
			# 		best_eval_valid = eval_valid
			# 		epochs_without_impr = 0
			# 		print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Best valid measure:'+str(eval_valid))
			# 	else:
			# 		epochs_without_impr+=1
			# 		if epochs_without_impr>self.args.early_stop_patience:
			# 			print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Early stop.')
			# 			break
			assert len(self.splitter.test)>0, \
                'there\'s no test samples'
			if len(self.splitter.test)>0 and e>=self.args.eval_after_epochs-1 and self.rank == 0:
				test_data = iter(copy.deepcopy(test_data_load))
				_, precision, recall, f1, acc = self.run_epoch(test_data, e, 'TEST', grad = False)

				if self.args.distributed:
					print("[{},{}] | Epoch:{} ended {}/{} at {} on {} | loss: {} precision: {} recall: {}, f1: {}, acc: {}, time cost: {}".format(
						os.getpid(), self.rank, e, self.rank+1, self.DIST_DEFAULT_WORLD_SIZE, self.DIST_DEFAULT_INIT_METHOD, self.device, loss,
						precision, recall, f1, acc, train_epoch_time_end - train_epoch_time_start))
				else:
					print(f"[{os.getpid()}] Epoch-{e} ended on {self.device}")

				# save the output
				Precision.append(precision)
				Recall.append(recall)
				F1.append(f1)
				if self.args.save_node_embeddings:
					self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, log_file+'_train_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.dev_idx, log_file+'_valid_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, log_file+'_test_nodeembs.csv.gz')

		dataframe = pd.DataFrame(time_spend, columns=['X'])
		dataframe = pd.concat([dataframe, pd.DataFrame(Loss,columns=['Y'])],axis=1)
		dataframe = pd.concat([dataframe, pd.DataFrame(Precision,columns=['Z'])],axis=1)
		dataframe = pd.concat([dataframe, pd.DataFrame(Recall,columns=['P'])],axis=1)
		dataframe = pd.concat([dataframe, pd.DataFrame(F1,columns=['Q'])],axis=1)
		dataframe.to_csv(f"../result/{self.args.partition}_{self.args.data}_{self.DIST_DEFAULT_WORLD_SIZE}.csv",header = False,index=False,sep=',')

	def run_epoch(self, split, epoch, set_name, grad):
		Loss = []
		torch.set_grad_enabled(grad)

		for s in split:  # 一次一个训练样本，每个训练样本（某一时刻的图）会生成一个时序图，s为时序图
			if self.tasker.is_static:
				s = self.prepare_static_sample(s)
			else:
				s = self.prepare_sample(s)  #将稀疏矩阵转为稠密矩阵，用来计算
			print(self.rank,': prepare sample complete!')
			predictions, nodes_embs = self.predict(self.gcn, s.hist_adj_list,      # s.hist_adj_list 存储时序图每个时刻下的邻接矩阵
												   s.hist_ndFeats_list,            # s.hist_ndFeats_list 存储时序图每个时刻下的节点特征矩阵
												   s.label_sp,              # s.label_sp['idx] 训练节点序号
												   s.node_mask_list)
			print(self.rank,': forward complete!')
			# back proporgation
			labels = []
			for time in range (len(s.hist_adj_list)):
				labels.append(s.label_sp[time]['vals'])
			predictions = torch.cat(predictions, dim=0)
			labels = torch.cat(labels, dim=0)
			loss = self.comp_loss(predictions,labels)
			Loss.append(loss)
			print(self.rank,': compute loss complete!')
			# release the GPU
			# for i, adj in enumerate(s.hist_adj_list):
			# 	s.hist_adj_list[i].to('cpu')
			# 	s.hist_ndFeats_list[i].to('cpu')
			# 	s.node_mask_list[i].to('cpu')
			# predictions.cpu()
			# s.label_sp['idx'].cpu()
			# s.label_sp['vals'].cpu()

			# 测试集上计算precision，recall和f1
			if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred' and self.rank == 0:
				precision, recall, f1, acc = self.compute_acc(predictions, labels)

		# average training loss
		if set_name == 'TRAIN':
			loss = sum(Loss)
			if grad:
				self.optim_step(loss)
			print(self.rank,': backward complete!')

		torch.set_grad_enabled(True)
		if set_name=='TEST':
			return nodes_embs, precision, recall, f1, acc
		else:
			return loss, nodes_embs

	def predict(self,gcn,hist_adj_list,hist_ndFeats_list,label_sp,mask_list):
		gather_prediction_list = []
		# 返回最后一时刻的图节点embeddings
		nodes_embs = gcn(hist_adj_list,
							  hist_ndFeats_list,
							  mask_list)

		predict_batch_size = 128

		# print(nodes_embs,node_indices)
		for time in range (len(hist_adj_list)):
			gather_predictions=[]
			node_indices = label_sp[time]['idx']
			for i in range(1 +(node_indices.size(1)//predict_batch_size)):
				cls_input = self.gather_node_embs(nodes_embs[time], node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])  # 获取一个batch的边embedding
				predictions = self.classifier(cls_input)
				gather_predictions.append(predictions)
			gather_prediction_list.append(torch.cat(gather_predictions, dim=0))
			# gather_prediction_list.append(gather_predictions)
		return gather_prediction_list, nodes_embs

	def gather_node_embs(self,nodes_embs,node_indices):
		cls_input = []

		for node_set in node_indices:
			cls_input.append(nodes_embs[node_set])
		return torch.cat(cls_input,dim = 1)  # cls_input列表有两个元素，第一个元素存储边起始点的特征矩阵，第二个元素存储边终点的特征矩阵，使用cat函数将起始点和终点的特征拼接

	def optim_step(self,loss):
		self.tr_step += 1
		if self.tr_step % self.args.steps_accum_gradients == 0:
			self.gcn_opt.zero_grad()
			self.classifier_opt.zero_grad()
			time_start = time.time()
			loss.backward()
			time_end = time.time()
			print(self.rank,': compute gradients! Time costs:', time_end - time_start)
			self.gcn_opt.step()
			self.classifier_opt.step()



	def prepare_sample(self,sample):
		sample = u.Namespace(sample)
		for i,adj in enumerate(sample.hist_adj_list):
			adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])  # tensor稀疏矩阵转稠密矩阵
			sample.hist_adj_list[i] = adj.to(self.device)

			nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])  # 稀疏的点的特征矩阵转稠密矩阵
			# print(nodes)
			# if feature partition
			if self.args.partition == 'feature':
				nodes = nodes.to_dense()  # to dense matrix
				if self.rank != self.DIST_DEFAULT_WORLD_SIZE - 1:
					nodes = nodes[:,self.rank*self.feature_per_node:(self.rank+1)*self.feature_per_node]
				else:
					nodes = nodes[:,self.rank*self.feature_per_node:]

			sample.hist_ndFeats_list[i] = nodes.to(self.device)
			# print(sample.hist_ndFeats_list[i])
			node_mask = sample.node_mask_list[i]
			sample.node_mask_list[i] = node_mask.to(self.device).t() #transposed to have same dimensions as scorer

		label_sp = self.ignore_batch_dim(sample.label_sp)
		for i in range (len(label_sp)):
			if self.args.task in ["link_pred", "edge_cls"]:
				# 原始的label稀疏矩阵为[[source,target], [sorce,target]],需要转换为[[source_set], [target_Set]]方便获取对应点的特征
				label_sp[i]['idx'] = label_sp[i]['idx'].to(self.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
			else:
				label_sp[i]['idx'] = label_sp[i]['idx'].to(self.device)

			label_sp[i]['vals'] = label_sp[i]['vals'].type(torch.long).to(self.device)
		sample.label_sp = label_sp

		return sample

	def prepare_static_sample(self,sample):
		sample = u.Namespace(sample)
		sample.hist_adj_list = self.hist_adj_list
		sample.hist_ndFeats_list = self.hist_ndFeats_list
		label_sp = {}
		label_sp['idx'] =  [sample.idx]
		label_sp['vals'] = sample.label
		sample.label_sp = label_sp
		return sample

	def ignore_batch_dim(self,adj):
		for i in range (len(adj)):
			if self.args.task in ["link_pred", "edge_cls"]:
				adj[i]['idx'] = adj[i]['idx'][0]
			adj[i]['vals'] = adj[i]['vals'][0]
		return adj

	def save_node_embs_csv(self, nodes_embs, indexes, file_name):
		csv_node_embs = []
		for node_id in indexes:
			orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

			csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

		pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
		#print ('Node embs saved in',file_name)

	def compute_auc(self, predictions_pos, predictions_neg, labels):
		Pre = torch.cat([predictions_pos,predictions_neg]).cpu().numpy()
		return roc_auc_score(labels.cpu().numpy(),Pre)

	def compute_acc(self, predictions, labels):
		predicted_classes = predictions.argmax(dim=1)
		precision = precision_score(labels.cpu(), predicted_classes.cpu(), average='binary')
		recall = recall_score(labels.cpu(), predicted_classes.cpu(), average='binary')
		f1 = f1_score(labels.cpu(), predicted_classes.cpu(), average='binary')
		acc = accuracy_score(labels.cpu(), predicted_classes.cpu())
		return precision, recall, f1, acc