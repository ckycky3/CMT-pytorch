import numpy as np
import torch
from collections import defaultdict
import random
from tqdm import tqdm
import pickle

from utils import logger
from utils.metrics import cal_metrics
from utils.utils import *
from dataset import collate_fn

x_fontdict = {'fontsize': 6,
             'verticalalignment': 'top',
             'horizontalalignment': 'left',
             'rotation': 'vertical',
             'rotation_mode': 'anchor'}
y_fontdict = {'fontsize': 6}


class BaseTrainer:
    def __init__(self, asset_path, model, criterion, optimizer,
                 train_loader, eval_loader, test_loader,
                 config):
        self.asset_path = asset_path
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.verbose = config['verbose']

        # metrics
        self.metrics = config['metrics']

        # dataloader
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader

        self.loading_epoch = 1
        self.current_step = 0
        self.losses = defaultdict(list)

    def _step(self, loss, **kwargs):
        raise NotImplementedError()

    def _epoch(self, epoch, mode, **kwargs):
        raise NotImplementedError()

    def train(self, **kwargs):
        raise NotImplementedError()

    def load_model(self, restore_epoch):
        if os.path.isfile(os.path.join(self.asset_path, 'model', 'checkpoint_%d.pth.tar' % restore_epoch)):
            checkpoint = torch.load(os.path.join(self.asset_path, 'model', 'checkpoint_%d.pth.tar' % restore_epoch))
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.current_step = checkpoint['current_step']
            self.loading_epoch = checkpoint['epoch'] + 1
            logger.info("restore model with %d epoch" % restore_epoch)
        else:
            logger.info("no checkpoint with %d epoch" % restore_epoch)

    def save_model(self, epoch, current_step):
        logger.info('saving model, Epoch %d, step %d' % (epoch, current_step))
        model_save_path = os.path.join(self.asset_path, 'model', 'checkpoint_%d.pth.tar' % epoch)
        state_dict = {'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'current_step': current_step,
                      'epoch': epoch}
        torch.save(state_dict, model_save_path)

    def adjust_learning_rate(self, factor=.5, min_lr=0.000001):
        losses = self.losses['eval']
        if len(losses) > 4 and losses[-1] > np.mean(losses[-4:-1]):
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = max(old_lr * factor, min_lr)
                param_group['lr'] = new_lr
                logger.info('adjusting learning rate from %.6f to %.6f' % (old_lr, new_lr))


class C2MTtrainer(BaseTrainer):
    def __init__(self, asset_path, model, criterion, optimizer,
                 train_loader, eval_loader, test_loader,
                 config):
        super(C2MTtrainer, self).__init__(asset_path, model, criterion, optimizer,
                                          train_loader, eval_loader, test_loader,
                                          config)
        # for logging
        self.losses = defaultdict(list)
        self.tf_logger = get_tflogger(asset_path)

    def train(self, **kwargs):
        # load model if exists
        self.load_model(kwargs["restore_epoch"])

        # start training
        for epoch in range(self.loading_epoch, self.config['max_epoch']):
            logger.info("")
            logger.info("%d epoch" % epoch)

            # train epoch
            logger.info("==========train %d epoch==========" % epoch)
            self._epoch(epoch, mode='train')

            # valid epoch and sampling
            with torch.no_grad():
                logger.info("==========valid %d epoch==========" % epoch)
                self._epoch(epoch, mode='eval')
                if epoch > self.loading_epoch and epoch % 10 == 0:
                    self.save_model(epoch, self.current_step)
                    self._sampling(epoch)

    def _step(self, loss, **kwargs):
        # back-propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.current_step += 1

    def _epoch(self, epoch, mode, **kwargs):
        # enable eval mode
        if mode == 'train':
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.eval_loader

        results = defaultdict(float)
        total_pitch_loss = 0.
        total_rhythm_loss = 0.
        num_total = 0
        for i, data in enumerate(loader):
            # preprocessing and forwarding
            result_dict = self.model(data['rhythm'], data['pitch'][:, :-1],
                                     data['chord'], False)
            rhythm_out = result_dict['rhythm']
            rhythm_out = rhythm_out.view(-1, rhythm_out.size(-1))
            pitch_out = result_dict['pitch']
            pitch_out = pitch_out.view(-1, pitch_out.size(-1))
            num_total += pitch_out[:, :, 0].numel()

            # get loss & metric(accuracy)
            rhythm_criterion = self.criterion[0]
            pitch_criterion = self.criterion[1]

            rhythm_loss = rhythm_criterion(rhythm_out, data['rhythm'][:, 1:].contiguous().view(-1))
            total_rhythm_loss += rhythm_loss.item()
            pitch_loss = pitch_criterion(pitch_out, data['pitch'][:, 1:].contiguous().view(-1))
            total_pitch_loss += pitch_loss.item()
            loss = pitch_loss + rhythm_loss

            result = dict()
            result.update(cal_metrics(pitch_out, data['pitch'][:, 1:].contiguous().view(-1),
                                      self.metrics, mode, name='pitch'))
            result.update(cal_metrics(rhythm_out, data['rhythm'][:, 1:].contiguous().view(-1),
                                      self.metrics, mode, name='rhythm'))

            for key, val in result.items():
                results[key] += val

            # do training operations
            if mode == 'train':
                self._step(loss)
                # self._step(rhythm_loss)
                if self.verbose and self.current_step % 100 == 0:
                    logger.info("%d training steps" % self.current_step)
                    print_dict = {'nll': loss.item()}
                    print_dict.update({
                        'nll_pitch': pitch_loss,
                        'nll_rhythm': rhythm_loss})
                    print_result(print_dict, result)

        # logging epoch statistics and information
        results = {key: val / len(loader) for key, val in results.items()}
        footer = '/' + mode
        losses = {'nll' + footer: (total_rhythm_loss + total_pitch_loss) / len(loader),
                  'nll_pitch' + footer: total_pitch_loss / len(loader),
                  'nll_rhythm' + footer: total_rhythm_loss / len(loader)}
        print_result(losses, results)
        tensorboard_logging_result(self.tf_logger, epoch, losses)
        tensorboard_logging_result(self.tf_logger, epoch, results)

        self.losses[mode].append((total_rhythm_loss + total_pitch_loss) / len(loader))
        if mode == 'eval':
            self.adjust_learning_rate()

    def _sampling(self, epoch):
        self.model.eval()
        loader = self.test_loader
        asset_path = os.path.join(self.asset_path)

        indices = random.sample(range(len(loader.dataset)), self.config["num_sample"])
        batch = collate_fn([loader.dataset[i] for i in indices])
        prime = batch['pitch'][:, :self.config["num_prime"]]
        if isinstance(self.model, torch.nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        prime_rhythm = batch['rhythm'][:, :self.config["num_prime"]]
        result_dict = model.sampling(prime_rhythm, prime, batch['chord'],
                                     self.config["topk"], self.config['attention_map'])
        result_key = 'pitch'
        pitch_idx = result_dict[result_key].cpu().numpy()

        logger.info("==========sampling result of epoch %03d==========" % epoch)
        os.makedirs(os.path.join(asset_path, 'sampling_results', 'epoch_%03d' % epoch), exist_ok=True)

        for sample_id in range(pitch_idx.shape[0]):
            logger.info(("Sample %02d : " % sample_id) + str(pitch_idx[sample_id][self.config["num_prime"]:self.config["num_prime"]+20]))
            save_path = os.path.join(asset_path, 'sampling_results', 'epoch_%03d' % epoch,
                                     'epoch%03d_sample%02d.mid' % (epoch, sample_id))
            gt_pitch = batch['pitch'].cpu().numpy()
            gt_chord = batch['chord'][:, :-1].cpu().numpy()
            sample_dict = {'pitch': pitch_idx[sample_id],
                           'chord': chord_array_to_dict(gt_chord[sample_id])}
            sample_dict['rhythm'] = result_dict['rhythm'][sample_id].cpu().numpy()

            with open(save_path.replace('.mid', '.pkl'), 'wb') as f_samp:
                pickle.dump(sample_dict, f_samp)
            instruments = pitch_to_midi(pitch_idx[sample_id], gt_chord[sample_id], model.frame_per_bar, save_path)
            save_instruments_as_image(save_path.replace('.mid', '.jpg'), instruments,
                                      frame_per_bar=model.frame_per_bar,
                                      num_bars=(model.max_len // model.frame_per_bar))

            # save groundtruth
            logger.info(("Groundtruth %02d : " % sample_id) +
                        str(gt_pitch[sample_id, self.config["num_prime"]:self.config["num_prime"] + 20]))
            gt_path = os.path.join(asset_path, 'sampling_results', 'epoch_%03d' % epoch,
                                     'epoch%03d_groundtruth%02d.mid' % (epoch, sample_id))
            gt_dict = {'pitch': gt_pitch[sample_id, :-1],
                        'chord': chord_array_to_dict(gt_chord[sample_id])}
            gt_dict['rhythm'] = batch['rhythm'][sample_id, :-1].cpu().numpy()
            with open(gt_path.replace('.mid', '.pkl'), 'wb') as f_gt:
                pickle.dump(gt_dict, f_gt)
            gt_instruments = pitch_to_midi(gt_pitch[sample_id, :-1], gt_chord[sample_id], model.frame_per_bar, gt_path)
            save_instruments_as_image(gt_path.replace('.mid', '.jpg'), gt_instruments,
                                      frame_per_bar=model.frame_per_bar,
                                      num_bars=(model.max_len // model.frame_per_bar))