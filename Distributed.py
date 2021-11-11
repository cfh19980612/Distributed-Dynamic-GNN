import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data.distributed import DistributedSampler
import utils as u
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import splitter as sp
import time
import warnings

#datasets
# import bitcoin_dl as bc
# import elliptic_temporal_dl as ell_temp
# import uc_irv_mess_dl as ucim
# import auto_syst_dl as aus
# import reddit_dl as rdt
import sbm_dl as sbm
#taskers
import link_pred_tasker as lpt
# import edge_cls_tasker as ect
# import node_cls_tasker as nct

#models
import models as mls
import egcn_h
import egcn_o
import Cross_Entropy as ce
import trainer as tr

torch.multiprocessing.set_start_method('spawn',force=True)
warnings.filterwarnings('ignore')

DIST_DEFAULT_BACKEND = 'nccl'
DIST_DEFAULT_ADDR = 'localhost'
DIST_DEFAULT_PORT = '12344'
DIST_DEFAULT_INIT_METHOD = f'tcp://{DIST_DEFAULT_ADDR}:{DIST_DEFAULT_PORT}'
DIST_DEFAULT_WORLD_SIZE = 4

def build_random_hyper_params(args):
	if args.model == 'all':
		model_types = ['gcn', 'egcn_o', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_nogcn':
		model_types = ['egcn_o', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_noegcn3':
		model_types = ['gcn', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_nogruA':
		model_types = ['gcn', 'egcn_o', 'egcn_h', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
		args.model=model_types[args.rank]
	elif args.model == 'saveembs':
		model_types = ['gcn', 'gcn', 'skipgcn', 'skipgcn']
		args.model=model_types[args.rank]

# 创建数据集
def build_dataset(args, rankID):
	if args.data == 'bitcoinotc' or args.data == 'bitcoinalpha':
		if args.data == 'bitcoinotc':
			args.bitcoin_args = args.bitcoinotc_args
		elif args.data == 'bitcoinalpha':
			args.bitcoin_args = args.bitcoinalpha_args
		return bc.bitcoin_dataset(args)
	elif args.data == 'aml_sim':
		return aml.Aml_Dataset(args)
	elif args.data == 'elliptic':
		return ell.Elliptic_Dataset(args)
	elif args.data == 'elliptic_temporal':
		return ell_temp.Elliptic_Temporal_Dataset(args)
	elif args.data == 'uc_irv_mess':
		return ucim.Uc_Irvine_Message_Dataset(args)
	elif args.data == 'dbg':
		return dbg.dbg_dataset(args)
	elif args.data == 'colored_graph':
		return cg.Colored_Graph(args)
	elif args.data == 'autonomous_syst':
		return aus.Autonomous_Systems_Dataset(args)
	elif args.data == 'reddit':
		return rdt.Reddit_Dataset(args)
	elif args.data.startswith('sbm'):
		if args.data == 'sbm20':
			args.sbm_args = args.sbm20_args
		elif args.data == 'sbm50':
			args.sbm_args = args.sbm50_args
		return sbm.sbm_dataset(args, rankID)
	else:
		raise NotImplementedError('only arxiv has been implemented')

def build_tasker(args,dataset):
	if args.task == 'link_pred':
		return lpt.Link_Pred_Tasker(args,dataset)
	elif args.task == 'edge_cls':
		return ect.Edge_Cls_Tasker(args,dataset)
	elif args.task == 'node_cls':
		return nct.Node_Cls_Tasker(args,dataset)
	elif args.task == 'static_node_cls':
		return nct.Static_Node_Cls_Tasker(args,dataset)

	else:
		raise NotImplementedError('still need to implement the other tasks')

def build_gcn(args,tasker):
	gcn_args = u.Namespace(args.gcn_parameters)
	gcn_args.feats_per_node = tasker.feats_per_node
	if args.model == 'gcn':
		return mls.Sp_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	elif args.model == 'skipgcn':
		return mls.Sp_Skip_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	elif args.model == 'skipfeatsgcn':
		return mls.Sp_Skip_NodeFeats_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	else:
		assert args.num_hist_steps > 0, 'more than one step is necessary to train LSTM'
		if args.model == 'lstmA':
			return mls.Sp_GCN_LSTM_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'gruA':
			return mls.Sp_GCN_GRU_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'lstmB':
			return mls.Sp_GCN_LSTM_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'gruB':
			return mls.Sp_GCN_GRU_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'egcn':
			return egcn.EGCN(gcn_args, activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'egcn_h':
			return egcn_h.EGCN(gcn_args, tasker, activation = torch.nn.RReLU(), device = args.device)
		elif args.model == 'skipfeatsegcn_h':
			return egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device, skipfeats=True)
		elif args.model == 'egcn_o':
			return egcn_o.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
		else:
			raise NotImplementedError('need to finish modifying the models')

def build_classifier(args,tasker):
	if 'node_cls' == args.task or 'static_node_cls' == args.task:
		mult = 1
	else:
		mult = 2
	if 'gru' in args.model or 'lstm' in args.model:
		in_feats = args.gcn_parameters['lstm_l2_feats'] * mult
	elif args.model == 'skipfeatsgcn' or args.model == 'skipfeatsegcn_h':
		in_feats = (args.gcn_parameters['layer_2_feats'] + args.gcn_parameters['feats_per_node']) * mult
	else:
		in_feats = args.gcn_parameters['layer_2_feats'] * mult

	return mls.Classifier(args,in_features = in_feats, out_features = tasker.num_classes).to(args.device)

def worker(rank, args):

	# CPU or GPU?
	args.device='cpu'
	args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
	if args.use_cuda:
		args.device='cuda'
		print ("use CUDA:", args.use_cuda, "- device:", args.device)

	# 定义模型和数据集
	dataset = build_dataset(args, rank)
	tasker = build_tasker(args, dataset)
	splitter = sp.splitter(args, tasker, DIST_DEFAULT_WORLD_SIZE, rank)  #build the splitter
	gcn = build_gcn(args, tasker)  #build the models
	# print(gcn.parameters())
	# for p in gcn.parameters():
	# 	for i in p:
	# 		print(i)
	# for name, p in gcn.named_parameters(): print (name,p)
	# print( p for p in list(gcn.parameters()))
	# for key, param in gcn.named_parameters(): print(key, param)
	# print(gcn.state_dict()['GRCU_layers.0.GCN_init_weights'])
	# for name in gcn.state_dict():
	# 	print(name)
	print('Process {} gets graphs with {} timesteps'.format(os.getpid(), dataset.max_time.item()+1))
	# 分布式环境初始化，加入进程组，rank即为进程对应id

	if args.distributed:
		print(
            f"[{os.getpid()}] Initializing {rank}/{DIST_DEFAULT_WORLD_SIZE} at {DIST_DEFAULT_INIT_METHOD}"
        )
		dist.init_process_group(backend=DIST_DEFAULT_BACKEND,
                                init_method=DIST_DEFAULT_INIT_METHOD,
                                world_size=DIST_DEFAULT_WORLD_SIZE,
                                rank=rank)
		print(
            f"[{os.getpid()}] Computing {rank}/{DIST_DEFAULT_WORLD_SIZE} at {DIST_DEFAULT_INIT_METHOD}"
        )
		# 进程绑定GPU/CPU
		if args.device == 'cuda':
			torch.cuda.set_device(rank)
			gcn.cuda()
			args.device = gcn.device

		# 模型加载进设备
		gcn = nn.parallel.DistributedDataParallel(gcn, device_ids=[rank])

	# 非分布式，直接加载模型进GPU
	else: gcn = nn.DataParallel(gcn).cuda()

	classifier = build_classifier(args,tasker)
	#build a loss
	cross_entropy = ce.Cross_Entropy(args,dataset).to(args.device)

	#trainer
	trainer = tr.Trainer(args,
						 splitter = splitter,
						 gcn = gcn,
						 classifier = classifier,
						 comp_loss = cross_entropy,
						 dataset = dataset,
						 num_classes = tasker.num_classes,
						 device = args.device,
						 DIST_DEFAULT_WORLD_SIZE = DIST_DEFAULT_WORLD_SIZE,
						 DIST_DEFAULT_INIT_METHOD = DIST_DEFAULT_INIT_METHOD,
						 rank = rank)
	trainer.train()

# 定义分布式启动模块
def launch(args):
    tic = time.time()
    if args.distributed:
        mp.spawn(worker,
                 args=(args, ),
                 nprocs=DIST_DEFAULT_WORLD_SIZE,
                 join=True)
    else:
        worker(None, args)
    toc = time.time()
    print(f"Finished in {toc-tic:.2f}s")


# 主函数
if __name__ == '__main__':
	parser = u.create_parser() #定义超参数
	args = u.parse_args(parser)

	launch(args)







# # 读数据进程执行的代码
# def _write(q,Input):
#     print('Process(%s) completes training...' % os.getpid())
#     for x in Input:
#         q.put(x)
#         print('Put %s to queue...' % x)
#         time.sleep(random.random())
# # 读数据进程执行的代码:
# def _read(q):
#     print('Process(%s) is reading...' % os.getpid())
#     while True:
#         x = q.get(True)
#         print('Get %s from queue.' % x)

# # 本地训练
# def Local_training(args, q, model, dataset, tasker, splitter):
#     print('Process {} gets graphs with {} timesteps'.format(os.getpid(), dataset.max_time.item()+1))
#     classifier = build_classifier(args,tasker)
# 	#build a loss
#     cross_entropy = ce.Cross_Entropy(args,dataset).to(args.device)

# 	#trainer
#     trainer = tr.Trainer(args,
# 						 splitter = splitter,
# 						 gcn = model,
# 						 classifier = classifier,
# 						 comp_loss = cross_entropy,
# 						 dataset = dataset,
# 						 num_classes = tasker.num_classes)
#     trainer.train()
#     _write(q, "Hello")
# if __name__ == '__main__':
#     q = Queue()

#     # 1) 初始化
#     # torch.distributed.init_process_group(backend="nccl")
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('--local_rank', default=-1, type=int,
#     #                     help='node rank for distributed training')
#     # args = parser.parse_args()

#     # dist.init_process_group(backend='nccl')
#     # torch.cuda.set_device(args.local_rank)

#     # 2） 配置每个进程的gpu
#     # local_rank = torch.distributed.get_rank()
#     # torch.cuda.set_device(local_rank)
#     # device = torch.device("cuda", local_rank)

#     # dataset = RandomDataset(input_size, data_size) # 修改

#     # 3）使用DistributedSampler
#     # rand_loader = DataLoader(dataset=dataset,
#     #                         batch_size=batch_size,
#     #                         sampler=DistributedSampler(dataset))

#     # 4) 参数设定
#     parser = u.create_parser() #定义超参数
#     args = u.parse_args(parser)
#     args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
#     args.device='cpu'
#     if args.use_cuda:
#         args.device='cuda'
#     print ("use CUDA:", args.use_cuda, "- device:", args.device)

#     # 创建数据集
#     dataset1 = build_dataset(args, 0)
#     dataset2 = build_dataset(args, 1)
#     # print(dataset.max_time)

#     # 4) 定义模型和超参数
#     # args = build_random_hyper_params(args)
#     tasker1 = build_tasker(args,dataset1)
#     tasker2 = build_tasker(args,dataset2)
#     splitter1 = sp.splitter(args,tasker1)  #build the splitter
#     splitter2 = sp.splitter(args,tasker2)  #build the splitter
#     gcn1 = build_gcn(args, tasker1)  #build the models
#     gcn2 = build_gcn(args, tasker2)  #build the models
#     # 5) 封装之前要把模型移到对应的gpu
#     # gcn.to(device)

#     # 6） 启动进程开始训练
#     Client_1 = Process(target=Local_training, args=(args, q, gcn1, dataset1, tasker1, splitter1))
#     Client_2 = Process(target=Local_training, args=(args, q, gcn2, dataset2, tasker2, splitter1))
#     # _reader = Process(target=_read, args=(q,))
#     # 启动子进程_writer，写入:
#     Client_1.start()
#     Client_2.start()

#     # 等待_writer结束:
#     Client_1.join()
#     Client_2.join()
#     # _reader进程里是死循环，无法等待其结束，只能强行终止:
#     Client_2.terminate()