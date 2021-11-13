import torch
import utils as u
import logger
import time
import pandas as pd
import numpy as np
import os

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

		self.logger = logger.Logger(args, self.num_classes)

		self.init_optimizers(args)  #初始化优化器

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
		log_file = 'sbm50'

		# to csv
		time_spend = []
		loss = []
		Precision = []
		Recall = []
		F1 = []
		time_now = time.time()
		for e in range(self.args.num_epochs):
			print('Epoch: ',e)
			Loss, nodes_embs, precision, recall, f1 = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)  # 训练一个epoch，参数(训练集，epochID，‘Train’，梯度求解)
			time_end = time.time()
			time_spend.append(time_end-time_now)
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
				print('Start testing!')
				Loss, nodes_embs_test, precision, recall, f1 = self.run_epoch(self.splitter.test, e, 'TEST', grad = False)
				if self.args.distributed:
				# Namelist = []
				# for name in self.gcn.state_dict():
				# 	Namelist.append(name)
					print("[{}] | Epoch:{} ended {}/{} at {} on {} | loss: {} precision: {} recall: {}, f1: {}".format(
						os.getpid(), e, self.rank+1, self.DIST_DEFAULT_WORLD_SIZE, self.DIST_DEFAULT_INIT_METHOD, self.device, sum(Loss), precision, recall, f1))
				else:
					print(f"[{os.getpid()}] Epoch-{e} ended on {self.device}")

				# save the output
				loss.append(sum(Loss).item())
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
		dataframe.to_csv(f"./result/{self.args.data}.csv",header = False,index=False,sep=',')

	def run_epoch(self, split, epoch, set_name, grad):

		t0 = time.time()
		log_interval=999
		if set_name=='TEST':
			log_interval=1
		self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)
		Loss = []
		torch.set_grad_enabled(grad)
		for s in split:  # 一次一个训练样本，每个训练样本（某一时刻的图）会生成一个时序图，s为时序图
			if self.tasker.is_static:
				s = self.prepare_static_sample(s)
			else:
				s = self.prepare_sample(s)  #将稀疏矩阵转为稠密矩阵，用来计算

			predictions, nodes_embs = self.predict(self.gcn, s.hist_adj_list,
												   s.hist_ndFeats_list,
												   s.label_sp['idx'],
												   s.node_mask_list)

			loss = self.comp_loss(predictions,s.label_sp['vals'])
			# print(loss)
			if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred' and self.rank == 0:
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
			else:
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())
			if grad:
				self.optim_step(loss)
			Loss.append(loss)

		torch.set_grad_enabled(True)
		precision, recall, f1 = self.logger.log_epoch_done()

		return Loss, nodes_embs, precision, recall, f1

	def predict(self,gcn,hist_adj_list,hist_ndFeats_list,node_indices,mask_list):

		nodes_embs = gcn(hist_adj_list,
							  hist_ndFeats_list,
							  mask_list)

		predict_batch_size = 100000
		gather_predictions=[]
		for i in range(1 +(node_indices.size(1)//predict_batch_size)):
			cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
			predictions = self.classifier(cls_input)
			gather_predictions.append(predictions)
		gather_predictions=torch.cat(gather_predictions, dim=0)
		return gather_predictions, nodes_embs

	def gather_node_embs(self,nodes_embs,node_indices):
		cls_input = []

		for node_set in node_indices:
			cls_input.append(nodes_embs[node_set])
		return torch.cat(cls_input,dim = 1)

	def optim_step(self,loss):
		self.tr_step += 1
		loss.backward()

		if self.tr_step % self.args.steps_accum_gradients == 0:
			self.gcn_opt.step()
			self.classifier_opt.step()

			self.gcn_opt.zero_grad()
			self.classifier_opt.zero_grad()

	def prepare_sample(self,sample):
		sample = u.Namespace(sample)
		for i,adj in enumerate(sample.hist_adj_list):
			adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])  # tensor稀疏矩阵转稠密矩阵
			sample.hist_adj_list[i] = adj.to(self.device)

			nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])  # 稀疏的点的特征矩阵转稠密矩阵

			sample.hist_ndFeats_list[i] = nodes.to(self.device)
			node_mask = sample.node_mask_list[i]
			sample.node_mask_list[i] = node_mask.to(self.device).t() #transposed to have same dimensions as scorer

		label_sp = self.ignore_batch_dim(sample.label_sp)

		if self.args.task in ["link_pred", "edge_cls"]:
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