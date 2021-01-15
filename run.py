import os, random
import numpy as np
import torch
import torch.nn as nn
import argparse

from utils import logger
from utils.hparams import HParams
from utils.utils import make_save_dir, get_optimizer
from loss import FocalLoss
from dataset import get_loader
from model import ChordConditionedMelodyTransformer as CMT
from trainer import CMTtrainer

# hyperparameter - using argparse and parameter module
parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, help='experiment number',  default=0)
parser.add_argument('--gpu_index', '-g', type=int, default="0", help='GPU index')
parser.add_argument('--ngpu', type=int, default=4, help='0 = CPU.')
parser.add_argument('--optim_name', type=str, default='adam')
parser.add_argument('--restore_epoch', type=int, default=-1)
parser.add_argument('--load_rhythm', dest='load_rhythm', action='store_true')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:%d" % args.gpu_index if use_cuda else "cpu")

hparam_file = os.path.join(os.getcwd(), "hparams.yaml")

config = HParams.load(hparam_file)
data_config = config.data_io
model_config = config.model
exp_config = config.experiment

# configuration
asset_root = config.asset_root
asset_path = os.path.join(asset_root, 'idx%03d' % args.idx)
make_save_dir(asset_path, config)
logger.logging_verbosity(1)
logger.add_filehandler(os.path.join(asset_path, "log.txt"))

# seed
if args.seed > 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# get dataloader for training
logger.info("get loaders")
train_loader = get_loader(data_config, mode='train')
eval_loader = get_loader(data_config, mode='eval')
test_loader = get_loader(data_config, mode='test')

# build graph, criterion and optimizer
logger.info("build graph, criterion, optimizer and trainer")
model = CMT(**model_config)

if args.ngpu > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
model.to(device)

nll_criterion = nn.NLLLoss().to(device)
pitch_criterion = FocalLoss(gamma=2).to(device)
criterion = (nll_criterion, pitch_criterion)

if args.load_rhythm:
    rhythm_params = list()
    pitch_params = list()
    param_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    for name, param in param_model.named_parameters():
        if 'rhythm' in name:
            rhythm_params.append(param)
        else:
            pitch_params.append(param)
    rhythm_param_dict = {'params': rhythm_params, 'lr': 1e-6}
    pitch_param_dict = {'params': pitch_params}
    params = [rhythm_param_dict, pitch_param_dict]
else:
    params = model.parameters()

optimizer = get_optimizer(params, config.experiment['lr'],
                          config.optimizer, name=args.optim_name)

# get trainer
trainer = CMTtrainer(asset_path, model, criterion, optimizer,
                     train_loader, eval_loader, test_loader,
                     device, exp_config)

# start training - add additional train configuration
logger.info("start training")
trainer.train(restore_epoch=args.restore_epoch,
              load_rhythm=args.load_rhythm)
