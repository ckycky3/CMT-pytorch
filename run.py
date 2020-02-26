import os, random
import datetime
import numpy as np
import torch
import torch.nn as nn
import argparse

from utils import logger
from utils.hparams import HParams
from utils.utils import make_save_dir, get_optimizer
from losses import FocalLoss
from dataset import get_loader
from models import ChordLSTM_MusicTransformer, MusicTransformerCE
from models import ChordLSTM_BeatMusicTransformer, BeatMusicTransformer
from trainer import MTTrainer

# hyperparameter - using argparse and parameter module
parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, help='experiment number',  default=0)
parser.add_argument('--ws', type=str, help='machine number', default='10')
parser.add_argument('--gpu_index', '-g', type=int, default="0", help='GPU index')
parser.add_argument('--ngpu', type=int, default=4, help='0 = CPU.')
parser.add_argument('--optim_name', type=str, default='adam')
parser.add_argument('--lr_scheduler', type=str, default='exp')
parser.add_argument('--model', type=str, default='CLSTM_MT')
parser.add_argument('--pitch_loss', type=str, default=None)
parser.add_argument('--restore_epoch', type=int, default=-1)
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
asset_root = config.asset_root[args.ws]
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
train_loader = get_loader(data_config, mode='train', ws=args.ws)
eval_loader = get_loader(data_config, mode='eval', ws=args.ws)
test_loader = get_loader(data_config, mode='test', ws=args.ws)

# build graph, criterion and optimizer
logger.info("build graph, criterion, optimizer and trainer")
if args.model == 'CLSTM_MT':
    model = ChordLSTM_MusicTransformer(**model_config)
elif args.model == 'MTCE':
    model = MusicTransformerCE(**model_config)
elif args.model == 'CLSTM_BMT':
    model = ChordLSTM_BeatMusicTransformer(**model_config)
elif args.model == 'BMT':
    model = BeatMusicTransformer(**model_config)
else:
    raise NotImplementedError()

if args.ngpu > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
model.to(device)

nll_criterion = nn.NLLLoss().to(device)
if args.pitch_loss is not None:
    if args.pitch_loss == 'focal':
        pitch_criterion = FocalLoss(gamma=2).to(device)
    else:
        raise NotImplementedError()
    criterion = (nll_criterion, pitch_criterion)
else:
    criterion = nll_criterion

if 'beat_lr' in config.experiment.keys():
    beat_params = list()
    pitch_params = list()
    param_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    for name, param in param_model.named_parameters():
        if 'beat' in name:
            beat_params.append(param)
        else:
            pitch_params.append(param)
    beat_param_dict = {'params': beat_params, 'lr': config.experiment['beat_lr']}
    pitch_param_dict = {'params': pitch_params}
    params = [beat_param_dict, pitch_param_dict]
else:
    params = model.parameters()

optimizer = get_optimizer(params, config.experiment['lr'],
                          config.optimizer, name=args.optim_name)

# get trainer
trainer = MTTrainer(asset_path, model, criterion, optimizer,
                    train_loader, eval_loader, test_loader,
                    exp_config)

# start training - add additional train configuration
logger.info("start training")
trainer.train(restore_epoch=args.restore_epoch,
              lr_schedular=args.lr_scheduler)
