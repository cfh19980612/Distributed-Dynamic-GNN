import torch
import utils as u
import logger
import time
import pandas as pd
import numpy as np
import os

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

	def init_optimizers(self,args):
		# gcn网络优化器，即example中的EGCN网络
		params = self.gcn.parameters()
		self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
		self.gcn_opt.zero_grad()

		# 分类器网络优化器
		params = self.classifier.parameters()
		self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
		self.classifier_opt.zero_grad()

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
		loss = []
		Precision = []
		Recall = []
		F1 = []
		time_now = time.time()
		for e in range(self.args.num_epochs):
			train_epoch_time_start = time.time()
			# Namelist = []
			# for name in self.gcn.state_dict():
			# 	Namelist.append(name)
			# # print(Namelist)
			# print("Name:{} para:{}".format(Namelist[0],self.gcn.state_dict()[Namelist[0]]))
			Loss, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)  # 训练一个epoch，参数(训练集，epochID，‘Train’，梯度求解)
			train_epoch_time_end = time.time()
			print('Epoch time:',train_epoch_time_end - train_epoch_time_start)
			time_end = time.time()
			time_spend.append(time_end-time_now)
			loss.append(sum(Loss).item())
			# save the loss

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
				nodes_embs_test, precision, recall, f1, acc = self.run_epoch(self.splitter.test, e, 'TEST', grad = False)
				# precision = 1
				# recall = 1
				# f1 = 1
				# acc = '?'
				if self.args.distributed:
				# Namelist = []
				# for name in self.gcn.state_dict():
				# 	Namelist.append(name)
					# print("[{}] | Epoch:{} ended {}/{} at {} on {} | loss: {} precision: {} recall: {}, f1: {}, acc: {}".format(
					# 	os.getpid(), e, self.rank+1, self.DIST_DEFAULT_WORLD_SIZE, self.DIST_DEFAULT_INIT_METHOD, self.device, sum(Loss), precision, recall, f1, acc))
					print("[{}] | Epoch:{} ended {}/{} at {} on {} | loss: {} precision: {} recall: {}, f1: {}, acc: {}".format(
						os.getpid(), e, self.rank+1, self.DIST_DEFAULT_WORLD_SIZE, self.DIST_DEFAULT_INIT_METHOD, self.device, sum(Loss), precision, recall, f1, acc))
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
		dataframe = pd.concat([dataframe, pd.DataFrame(loss,columns=['Y'])],axis=1)
		dataframe = pd.concat([dataframe, pd.DataFrame(Precision,columns=['Z'])],axis=1)
		dataframe = pd.concat([dataframe, pd.DataFrame(Recall,columns=['P'])],axis=1)
		dataframe = pd.concat([dataframe, pd.DataFrame(F1,columns=['Q'])],axis=1)
		dataframe.to_csv(f"./result/{self.args.data}_{self.DIST_DEFAULT_WORLD_SIZE}.csv",header = False,index=False,sep=',')

	def run_epoch(self, split, epoch, set_name, grad):

		t0 = time.time()
		log_interval=999
		if set_name=='TEST':
			log_interval=1
		# self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)
		Loss = []
		torch.set_grad_enabled(grad)
		Acc = []
		frac = 0.1  # training set fraction
		time_cost_forward = 0
		time_cost_back = 0
		time_total_start = time.time()
		total_loss = []
		a = 0
		# print('current time 1: ', time.time())
		for s in split:  # 一次一个训练样本，每个训练样本（某一时刻的图）会生成一个时序图，s为时序图
			time_start = time.time()
			if self.tasker.is_static:
				s = self.prepare_static_sample(s)
				# print('current time 2: ', a, time.time())
			else:
				s = self.prepare_sample(s)  #将稀疏矩阵转为稠密矩阵，用来计算

			# for computing auc
			# # positive prediction
			# predictions_pos, nodes_embs = self.predict(self.gcn, s.hist_adj_list,      # s.hist_adj_list 存储时序图每个时刻下的邻接矩阵
			# 									   s.hist_ndFeats_list,            # s.hist_ndFeats_list 存储时序图每个时刻下的节点特征矩阵
			# 									   s.label_sp_pos['idx'],              # s.label_sp['idx] 训练节点序号
			# 									   s.node_mask_list)
			# # negative prediction
			# predictions_neg, nodes_embs = self.predict(self.gcn, s.hist_adj_list,      # s.hist_adj_list 存储时序图每个时刻下的邻接矩阵
			# 									   s.hist_ndFeats_list,            # s.hist_ndFeats_list 存储时序图每个时刻下的节点特征矩阵
			# 									   s.label_sp_neg['idx'],              # s.label_sp['idx] 训练节点序号
			# 									   s.node_mask_list)
			# # print(predictions_pos.size())
			# # compute loss
			# scores = torch.cat([predictions_pos.squeeze(1), predictions_neg.squeeze(1)])
			# labels = torch.cat([s.label_sp_pos['vals'], s.label_sp_neg['vals']]).type_as(scores)
			# loss = F.binary_cross_entropy_with_logits(scores, labels)
			# print(predictions.size(0), s.label_sp['idx'].size(1), s.label_sp['vals'].size(0))

			predictions, nodes_embs = self.predict(self.gcn, s.hist_adj_list,      # s.hist_adj_list 存储时序图每个时刻下的邻接矩阵
												   s.hist_ndFeats_list,            # s.hist_ndFeats_list 存储时序图每个时刻下的节点特征矩阵
												   s.label_sp['idx'],              # s.label_sp['idx] 训练节点序号
												   s.node_mask_list)
			# print('current time 3: ', a, time.time())
			loss = self.comp_loss(predictions,s.label_sp['vals'])
			# print('current time 4: ', a, time.time())

			time_end = time.time()
			time_cost_forward += time_end - time_start

			# release the GPU
			for i, adj in enumerate(s.hist_adj_list):
				print(s.hist_adj_list[i].device)
				s.hist_adj_list[i].to('cpu')
				s.hist_ndFeats_list[i].to('cpu')
				s.node_mask_list[i].to('cpu')
			# print('current time 5: ', a, time.time())
			predictions.cpu()
			s.label_sp['idx'].cpu()
			s.label_sp['vals'].cpu()
			# print('current time 6: ', a, time.time())
			#acc = self.compute_acc(predictions, s.label_sp['vals'])
			# print('Prediction:{}, Label:{}, acc:{}'.format(predictions.size(0),s.label_sp['vals'].size(0), acc))

			# 测试集上计算precision，recall和f1
			if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred' and self.rank == 0:
				# self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
				# precision, recall, f1 = self.compute_acc(predictions, s.label_sp['vals'])
				precision, recall, f1, acc = self.compute_acc(predictions, s.label_sp['vals'])

			# print('processing one graph time: ',time_end - time_start)
			# else:
			# 	self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())
			Loss.append(loss)
			# print('current time 7: ', a, time.time())
		loss = sum(Loss)/len(Loss)
		# backward
		time_start_back = time.time()
		# print('current time 8: ', time.time())
		if grad:
			self.optim_step(loss)
		# print('current time 9: ', time.time())
		time_end_back = time.time()

		time_cost_back += time_end_back - time_start_back


		time_total_end = time.time()

		print('forwarding graphs: ',time_cost_forward)
		print('backwarding graphs: ',time_cost_back)
		print('processing graphs: ',time_total_end - time_total_start)
		time_other_start = time.time()
		torch.set_grad_enabled(True)
		# print('current time 10: ', time.time())
		time_other_end = time.time()

		print('other time: ', time_other_end - time_other_start)
		# precision, recall, f1 = self.logger.log_epoch_done()

		if set_name=='TEST':
			# precision, recall, f1 = self.compute_acc()
			return nodes_embs, precision, recall, f1, acc
		else:
			return Loss, nodes_embs

	def predict(self,gcn,hist_adj_list,hist_ndFeats_list,node_indices,mask_list):

		# 返回最后一时刻的图节点embeddings
		nodes_embs = gcn(hist_adj_list,
							  hist_ndFeats_list,
							  mask_list)

		predict_batch_size = 128
		gather_predictions=[]

		# print(nodes_embs,node_indices)
		for i in range(1 +(node_indices.size(1)//predict_batch_size)):
			cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])  # 生成一个batch的边embedding

			predictions = self.classifier(cls_input)
			# if i == 0:
			# 	print(i,cls_input,predictions)
			gather_predictions.append(predictions)
		gather_predictions=torch.cat(gather_predictions, dim=0)
		return gather_predictions, nodes_embs

	def gather_node_embs(self,nodes_embs,node_indices):
		cls_input = []

		for node_set in node_indices:
			cls_input.append(nodes_embs[node_set])
		return torch.cat(cls_input,dim = 1)  # cls_input列表有两个元素，第一个元素存储边起始点的特征矩阵，第二个元素存储边终点的特征矩阵，使用cat函数将起始点和终点的特征拼接

	def optim_step(self,loss):
		self.tr_step += 1
		loss.backward()

		if self.tr_step % self.args.steps_accum_gradients == 0:
			self.gcn_opt.zero_grad()
			self.classifier_opt.zero_grad()

			self.gcn_opt.step()
			self.classifier_opt.step()

	def prepare_sample(self,sample):
		sample = u.Namespace(sample)
		for i,adj in enumerate(sample.hist_adj_list):
			adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])  # tensor稀疏矩阵转稠密矩阵
			sample.hist_adj_list[i] = adj.to(self.device)

			nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])  # 稀疏的点的特征矩阵转稠密矩阵

			sample.hist_ndFeats_list[i] = nodes.to(self.device)
			node_mask = sample.node_mask_list[i]
			sample.node_mask_list[i] = node_mask.to(self.device).t() #transposed to have same dimensions as scorer

		# # prepare positive edges'label
		# label_sp_pos = self.ignore_batch_dim(sample.label_sp_pos)
		# if self.args.task in ["link_pred", "edge_cls"]:
		# 	# 原始的label稀疏矩阵为[[source,target], [sorce,target]],需要转换为[[source_set], [target_Set]]方便获取对应点的特征
		# 	label_sp_pos['idx'] = label_sp_pos['idx'].to(self.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
		# else:
		# 	label_sp_pos['idx'] = label_sp_pos['idx'].to(self.device)

		# label_sp_pos['vals'] = label_sp_pos['vals'].type(torch.long).to(self.device)
		# sample.label_sp_pos = label_sp_pos

		# # prepare negative edges'label
		# label_sp_neg = self.ignore_batch_dim(sample.label_sp_neg)
		# if self.args.task in ["link_pred", "edge_cls"]:
		# 	# 原始的label稀疏矩阵为[[source,target], [sorce,target]],需要转换为[[source_set], [target_Set]]方便获取对应点的特征
		# 	label_sp_neg['idx'] = label_sp_neg['idx'].to(self.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
		# else:
		# 	label_sp_neg['idx'] = label_sp_neg['idx'].to(self.device)

		# label_sp_neg['vals'] = label_sp_neg['vals'].type(torch.long).to(self.device)
		# sample.label_sp_neg = label_sp_neg

		label_sp = self.ignore_batch_dim(sample.label_sp)
		if self.args.task in ["link_pred", "edge_cls"]:
			# 原始的label稀疏矩阵为[[source,target], [sorce,target]],需要转换为[[source_set], [target_Set]]方便获取对应点的特征
			label_sp['idx'] = label_sp['idx'].to(self.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
		else:
			label_sp['idx'] = label_sp['idx'].to(self.device)

		label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.device)
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
		if self.args.task in ["link_pred", "edge_cls"]:
			adj['idx'] = adj['idx'][0]
		adj['vals'] = adj['vals'][0]
		return adj

	def save_node_embs_csv(self, nodes_embs, indexes, file_name):
		csv_node_embs = []
		for node_id in indexes:
			orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

			csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

		pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
		#print ('Node embs saved in',file_name)

	def compute_auc(self, predictions_pos, predictions_neg, labels):
		# acc = 0
		# for i in range(label.size(0)):
		# 	if prediction[1].data.max(0)[1] == label[i].data:
		# 		acc += 1
		# # print(acc)
		# return float(acc/label.size(0))
		# pos = prediction[:5000,0:1]
		# neg = prediction[5000:,1:2]
		Pre = torch.cat([predictions_pos,predictions_neg]).cpu().numpy()
		return roc_auc_score(labels.cpu().numpy(),Pre)

	def compute_acc(self, predictions, labels):
		predicted_classes = predictions.argmax(dim=1)
		precision = precision_score(labels.cpu(), predicted_classes.cpu(), average='binary')
		recall = recall_score(labels.cpu(), predicted_classes.cpu(), average='binary')
		f1 = f1_score(labels.cpu(), predicted_classes.cpu(), average='binary')
		acc = accuracy_score(labels.cpu(), predicted_classes.cpu())
		return precision, recall, f1, acc

	# def log_info():
	# 	for cl in range(self.num_classes):
	# 		conf_mat_tp[cl]=0
	# 		conf_mat_fn[cl]=0
	# 		conf_mat_fp[cl]=0
	# 		for k in eval_k_list:
	# 			conf_mat_tp_at_k[k][cl]=0
	# 			conf_mat_fn_at_k[k][cl]=0
	# 			conf_mat_fp_at_k[k][cl]=0
	# 	if set == "TEST":
    #         conf_mat_tp_list = {}
    #         conf_mat_fn_list = {}
    #         conf_mat_fp_list = {}
    #         for cl in range(num_classes):
    #             conf_mat_tp_list[cl]=[]
    #             conf_mat_fn_list[cl]=[]
    #             conf_mat_fp_list[cl]=[]

	# def log_info(set):

	# 	precision, recall, f1 = calc_microavg_eval_measures(conf_mat_tp, conf_mat_fn, conf_mat_fp)
	# 	print(set+' measures microavg - precision %0.4f - recall %0.4f - f1 %0.4f ' % (precision,recall,f1))

	# def calc_microavg_eval_measures(tp, fn, fp):
	# 	#ALDO
    #     if type(tp) is dict:
    #         tp_sum = tp[class_id].item()
    #         fn_sum = fn[class_id].item()
    #         fp_sum = fp[class_id].item()
    #     else:
    #         tp_sum = tp.item()
    #         fn_sum = fn.item()
    #         fp_sum = fp.item()
    #     ########
    #     if tp_sum==0:
    #         return 0,0,0

    #     p = tp_sum*1.0 / (tp_sum+fp_sum)
    #     r = tp_sum*1.0 / (tp_sum+fn_sum)
    #     if (p+r)>0:
    #         f1 = 2.0 * (p*r) / (p+r)
    #     else:
    #         f1 = 0
    #     return p, r, f1