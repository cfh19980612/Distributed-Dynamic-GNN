import torch
import torch.nn as nn
import utils as u
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import splitter as sp
import time
import warnings
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.nn import RemoteModule
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.rpc as rpc
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import sys
sys.path.append('../')
#datasets
from data_process import bitcoin_dl as bc
# import elliptic_temporal_dl as ell_temp
# import uc_irv_mess_dl as ucim
from data_process import auto_syst_dl as aus
# import reddit_dl as rdt
from data_process import sbm_dl as sbm
from data_process import epinion_dl as ep
from data_process import yt_dl as yt

#taskers
from tasker import link_pred_tasker as lpt
# import edge_cls_tasker as ect
# import node_cls_tasker as nct

#models
import models as mls
import egcn_h
from model import egcn_o
from model import egcn_h_fp
import Cross_Entropy as ce

import trainer as tr

torch.multiprocessing.set_start_method('spawn',force=True)
warnings.filterwarnings('ignore')

DIST_DEFAULT_BACKEND = 'nccl'
DIST_DEFAULT_ADDR = 'localhost'
DIST_DEFAULT_PORT = '12345'
DIST_DEFAULT_INIT_METHOD = f'tcp://{DIST_DEFAULT_ADDR}:{DIST_DEFAULT_PORT}'
DIST_DEFAULT_WORLD_SIZE = 2
rpc_backend_options = TensorPipeRpcBackendOptions()
rpc_backend_options.init_method = "tcp://localhost:12349"

GCN = [None for i in range (DIST_DEFAULT_WORLD_SIZE)]
Remote_Module = [None for i in range (DIST_DEFAULT_WORLD_SIZE - 1)]

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

# ???????????????
def build_dataset(args):
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
		return sbm.sbm_dataset(args)
	elif args.data == 'epinions':
		return ep.epinion_dataset(args)
	elif args.data == 'youtube':
		return yt.youtube_dataset(args)
	else:
		raise NotImplementedError('only arxiv has been implemented')

# ????????????
def build_tasker(args,dataset):
	if args.task == 'link_pred':
		return lpt.Link_Pred_Tasker(args,dataset,DIST_DEFAULT_WORLD_SIZE)
	elif args.task == 'edge_cls':
		return ect.Edge_Cls_Tasker(args,dataset)
	elif args.task == 'node_cls':
		return nct.Node_Cls_Tasker(args,dataset)
	elif args.task == 'static_node_cls':
		return nct.Static_Node_Cls_Tasker(args,dataset)

	else:
		raise NotImplementedError('still need to implement the other tasks')

# ??????gcn
def build_gcn(args,tasker,rank,remote_module = None):
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
			if args.partition == 'feature':
				return egcn_h_fp.EGCN(gcn_args, tasker, activation = torch.nn.RReLU())
			else:
				return egcn_h.EGCN(gcn_args, tasker, rank, remote_module, activation = torch.nn.RReLU(), device = args.device)
		elif args.model == 'skipfeatsegcn_h':
			return egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device, skipfeats=True)
		elif args.model == 'egcn_o':
			return egcn_o.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
		else:
			raise NotImplementedError('need to finish modifying the models')

# ???????????????
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
		in_feats = args.gcn_parameters['layer_2_feats'] * mult  # mult=2 ???????????????????????????????????????????????????????????????????????????100*2??? 100 ???egcn?????????embedding??????

	return mls.Classifier(args,in_features = in_feats, out_features = tasker.num_classes).to(args.device)

def build_gcn_fp(args,tasker):
	return mls.gcn(tasker.feats_per_node, args.gcn_parameters['layer_1_feats'])

class RNN_Send_Module(nn.Module):
	def __init__(self, gcn):
		self.gcn = gcn
	def forward(self, layer):
		return self.gcn.GCN_list[layer].cpu()

def worker(rank, args, dataset, tasker):
	# build splitter, use the rankid to split the dataset
	splitter = sp.splitter(args, tasker, DIST_DEFAULT_WORLD_SIZE, rank)
	print('Trainer{} splitter complete!'.format(rank))

	trainer_name = 'trainer{}'.format(rank)

	if args.distributed:
		# CPU or GPU
		args.device='cpu'
		args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
		if args.use_cuda:
			args.device='cuda'
			print('Trainer{} uses CUDA: {}, - device: {}'.format(rank, args.use_cuda, args.device))

		# initialize the DDP group
		dist.init_process_group(
			backend=DIST_DEFAULT_BACKEND,
			init_method=DIST_DEFAULT_INIT_METHOD,
			world_size=DIST_DEFAULT_WORLD_SIZE,
			rank=rank
		)

		if rank == 0: # the first trainer
			print(trainer_name)
			# build gcn
			GCN[rank] = build_gcn(args, tasker, rank)
			# initialize the rpc group
			rpc_backend_options.set_devices(["cuda:0", "cuda:1"])
			rpc_backend_options.set_device_map('trainer1',{'cuda:0': 'cuda:1'})
			# options = TensorPipeRpcBackendOptions(
			# 			init_method= "tcp://localhost:12349",
			# 			device_maps={"trainer1": {0: 1}}
			# 			# maps worker0's cuda:0 to worker1's cuda:1
			# 		)
			rpc.init_rpc(
				trainer_name,
				rank = rank,
				world_size=DIST_DEFAULT_WORLD_SIZE,
				rpc_backend_options=rpc_backend_options,
			)
			# print('rank',Remote_Module)

		elif DIST_DEFAULT_WORLD_SIZE > 1 and rank == DIST_DEFAULT_WORLD_SIZE -1:  # the final trainer has no remote module output
			print(trainer_name)
			# build gcn
			# initialize the rpc group
			# rpc_backend_options.set_device_map('trainer0',{rank: rank - 1})
			# rpc_backend_options.set_devices(["cuda:1"])
			rpc.init_rpc(
				trainer_name,
				rank = rank,
				world_size=DIST_DEFAULT_WORLD_SIZE,
				rpc_backend_options=rpc_backend_options,
			)
			if DIST_DEFAULT_WORLD_SIZE > 1:
				# build remote module output
				Remote_Module[rank-1] = RemoteModule(
								trainer_name,
								module_cls = RNN_Send_Module,
								args=(GCN[rank-1]),
								)
			# print('rank',Remote_Module)
			GCN[rank] = build_gcn(args, tasker, rank, remote_module=Remote_Module[rank - 1])

		else:
			# build gcn
			# initialize the rpc group
			rpc.init_rpc(
				trainer_name,
				rank = rank,
				world_size=DIST_DEFAULT_WORLD_SIZE,
				rpc_backend_options=rpc_backend_options,
			)
			GCN[rank] = build_gcn(args, tasker, rank, remote_module=Remote_Module[rank - 1])
			# build remote module output
			Remote_Module[rank] = RemoteModule(
							trainer_name,
							RNN_Send_Module,
							args=(GCN[rank]),
							kwargs=None,
							)

		# bind gpu with rank
		if args.device == 'cuda':
			torch.cuda.set_device(rank)
			GCN[rank].cuda()
		# GCN[rank] = DDP(GCN[rank].cuda(rank), device_ids=[rank])

		# build the classifier
		classifier = build_classifier(args,tasker)
		#build a loss
		cross_entropy = ce.Cross_Entropy(args,dataset).to(GCN[rank].device)

		# build the trainer
		trainer = tr.Trainer(args,
					splitter = splitter,
					gcn = GCN[rank],
					classifier = classifier,
					comp_loss = cross_entropy,
					dataset = dataset,
					num_classes = tasker.num_classes,
					device = GCN[rank].device,
					DIST_DEFAULT_WORLD_SIZE = DIST_DEFAULT_WORLD_SIZE,
					DIST_DEFAULT_INIT_METHOD = DIST_DEFAULT_INIT_METHOD,
					rank = rank)
		trainer.train()

	# ????????????????????????????????????GPU
	else:
		gcn = nn.DataParallel(GCN[rank]).cuda()

		classifier = build_classifier(args,tasker)
		#build a loss
		cross_entropy = ce.Cross_Entropy(args,dataset).to(gcn.device)

		trainer = tr.Trainer(args,
				splitter = splitter,
				gcn = gcn,
				classifier = classifier,
				comp_loss = cross_entropy,
				dataset = dataset,
				num_classes = tasker.num_classes,
				device = gcn.device,
				DIST_DEFAULT_WORLD_SIZE = DIST_DEFAULT_WORLD_SIZE,
				DIST_DEFAULT_INIT_METHOD = DIST_DEFAULT_INIT_METHOD,
				rank = rank)
		trainer.train()


# ???????????????????????????
def main(args):

	# # define the master process
	# rpc_backend_options_master = TensorPipeRpcBackendOptions()
	# rpc_backend_options_master.init_method = "tcp://localhost:12346"
	# rpc.init_rpc(
	# 	'master',
	# 	rank = 0,
	# 	world_size=1,
	# 	rpc_backend_options=rpc_backend_options_master,
	# )
	# print('rpc complete')
	tic = time.time()

	# build dataset
	dataset = build_dataset(args)
	dataset.max_time = dataset.max_time - 40
	print('dataset complete!', dataset.num_nodes, dataset.max_time)
	# build task
	tasker = build_tasker(args, dataset)
	print('tasker complete!', tasker.feats_per_node)

	# build gcn and remote module
	# GCN = [None for i in range (DIST_DEFAULT_WORLD_SIZE)]
	# Remote_Module = [None for i in range (DIST_DEFAULT_WORLD_SIZE - 1)] # the last trainer has no remote output
	# for rank in range (DIST_DEFAULT_WORLD_SIZE):
	# 	trainer_name = 'trainer{}'.format(rank)

		# if rank == 0: # the first trainer
		# 	# build gcn
		# 	GCN[rank] = build_gcn(args, tasker, rank)
		# 	# build remote module output
		# 	Remote_Module[rank] = RemoteModule(
		# 					trainer_name,
		# 					RNN_Send_Module,
		# 					args=(GCN[rank]),
		# 					)

		# elif DIST_DEFAULT_WORLD_SIZE > 1 and rank == DIST_DEFAULT_WORLD_SIZE -1:  # the final trainer has no remote module output
		# 	# build gcn
		# 	GCN[rank] = build_gcn(args, tasker, rank, remote_module=Remote_Module[rank - 1])

		# else:
		# 	# build gcn
		# 	GCN[rank] = build_gcn(args, tasker, rank, remote_module=Remote_Module[rank - 1])
		# 	# build remote module output
		# 	Remote_Module[rank] = RemoteModule(
		# 					trainer_name,
		# 					RNN_Send_Module,
		# 					args=(GCN[rank]),
		# 					kwargs=None,
		# 					)

	if args.distributed:
		mp.spawn(worker,
					args=(args, dataset, tasker),
					nprocs=DIST_DEFAULT_WORLD_SIZE,
					join=True)
	else:
		worker(0, args, GCN, dataset, tasker)
	toc = time.time()
	print(f"Finished in {toc-tic:.2f}s")

# ?????????
if __name__ == '__main__':
	parser = u.create_parser() #???????????????
	args = u.parse_args(parser)

	main(args)