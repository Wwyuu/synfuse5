#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:04:48
LastEditTime: 2020-12-03 22:02:20
@Description: file content
'''
import os, importlib, torch, shutil
from solver.basesolver import BaseSolver
from utils.utils import maek_optimizer, make_loss, save_config, save_net_config, cpsnr, cssim, csam, cergas
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
from importlib import import_module
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils.config import save_yml
from research import import_algorithm
from research.losses.iso import unwrap_hrms
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Solver(BaseSolver):
    def __init__(self, cfg):
        super(Solver, self).__init__(cfg)
        self.init_epoch = self.cfg['schedule']
        
        net_name = self.cfg['algorithm'].lower()
        lib = import_algorithm(net_name)
        net = lib.Net

        assert (self.cfg['data']['n_colors']==4)
        self.model = net(
            num_channels=self.cfg['data']['n_colors'], 
            base_filter=64,  
            args = self.cfg
        )
        self.optimizer = maek_optimizer(self.cfg['schedule']['optimizer'], cfg, self.model.parameters())
        self.loss = make_loss(self.cfg['schedule']['loss'])
        # Cosine from 5e-4 down to 5e-8 over 500 epochs.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,500,5e-8)
        self.log_name = self.cfg['algorithm'] + '_' + str(self.cfg['data']['upsacle']) + '_' + str(self.timestamp)
        # save log
        self.writer = SummaryWriter(self.cfg['log_dir']+ str(self.log_name))
        save_net_config(self.log_name, self.model)
        save_yml(cfg, os.path.join(self.cfg['log_dir'] + str(self.log_name), 'config.yml'))
        save_config(self.log_name, 'Train dataset has {} images and {} batches.'.format(len(self.train_dataset), len(self.train_loader)))
        save_config(self.log_name, 'Val dataset has {} images and {} batches.'.format(len(self.val_dataset), len(self.val_loader)))
        save_config(self.log_name, 'Model parameters: '+ str(sum(param.numel() for param in self.model.parameters())))
        self._init_swanlab()

    def train(self): 
        with tqdm(total=len(self.train_loader), miniters=1,
                desc='Initial Training Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t:

            epoch_loss = 0
            for iteration, batch in enumerate(self.train_loader, 1):
                ms_image, lms_image, pan_image, bms_image, file = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), (batch[4])

                if self.cuda:
                    ms_image, lms_image, pan_image, bms_image = ms_image.cuda(self.gpu_ids[0]), lms_image.cuda(self.gpu_ids[0]), pan_image.cuda(self.gpu_ids[0]), bms_image.cuda(self.gpu_ids[0])

                self.optimizer.zero_grad()               
                self.model.train()

                y, aux = self._forward(lms_image, bms_image, pan_image)
                loops = None if aux is None else aux.get('loops')
                if torch.is_tensor(loops):
                    k = int(loops.shape[1])
                    exps = torch.arange(k, device=loops.device) - (k - 1)
                    weight = torch.pow(loops.new_tensor(2.0), exps)
                    weight = weight / weight.sum()
                    loss = loops.new_zeros(())
                    for i in range(loops.shape[1]):
                        loss = loss + weight[i] * self.loss(loops[:, i], ms_image)
                    loss = loss / (self.cfg['data']['batch_size'] * 2)
                else:
                    loss = self.loss(y, ms_image) / (self.cfg['data']['batch_size'] * 2)
                
                if self.cfg['schedule']['use_YCbCr']:
                    y_vgg = torch.unsqueeze(y[:,3,:,:], 1) 
                    y_vgg_3 = torch.cat([y_vgg, y_vgg, y_vgg], 1)
                    pan_image_3 = torch.cat([pan_image, pan_image, pan_image], 1)
                    vgg_loss = self.vggloss(y_vgg_3, pan_image_3)

                epoch_loss += loss.data
                #epoch_loss = epoch_loss + loss.data + vgg_loss.data
                t.set_postfix_str("Batch loss {:.4f}".format(loss.item()))
                t.update()

                loss.backward()
                # print("grad before clip:"+str(self.model.output_conv.conv.weight.grad))
                if self.cfg['schedule']['gclip'] > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg['schedule']['gclip']
                    )
                self.optimizer.step()
            self.scheduler.step()
            self.records['Loss'].append(epoch_loss / len(self.train_loader))
            self.writer.add_image('image1', ms_image[0][:3].clamp(0, 1), self.epoch)
            self.writer.add_image('image2', y[0][:3].clamp(0, 1), self.epoch)
            self.writer.add_image('image3', pan_image[0].clamp(0, 1), self.epoch)
            save_config(self.log_name, 'Initial Training Epoch {}: Loss={:.4f}'.format(self.epoch, self.records['Loss'][-1]))
            self.writer.add_scalar('Loss_epoch', self.records['Loss'][-1], self.epoch)
            self._swan_log({
                'train/loss': self.records['Loss'][-1],
                'train/lr': self.optimizer.param_groups[0]['lr'],
            })

    def eval(self):
        with tqdm(total=len(self.val_loader), miniters=1,
                desc='Val Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t1:
            psnr_list, ssim_list, sam_list, ergas_list = [], [], [], []
            scale = int(self.cfg['data'].get('upsacle', 4))
            for iteration, batch in enumerate(self.val_loader, 1):
                
                ms_image, lms_image, pan_image, bms_image, file = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), (batch[4])
                if self.cuda:
                    ms_image, lms_image, pan_image, bms_image = ms_image.cuda(self.gpu_ids[0]), lms_image.cuda(self.gpu_ids[0]), pan_image.cuda(self.gpu_ids[0]), bms_image.cuda(self.gpu_ids[0])

                self.model.eval()
                with torch.no_grad():
                    y, _ = self._forward(lms_image, bms_image, pan_image)
                    loss = self.loss(y, ms_image)

                batch_psnr, batch_ssim, batch_sam, batch_ergas = [], [], [], []
                for c in range(y.shape[0]):
                    if not self.cfg['data']['normalize']:
                        predict_y = (y[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                        ground_truth = (ms_image[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                    else:
                        predict_y = (y[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                        ground_truth = (ms_image[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                    # WINet/FAME val + py-tra/metrics: all bands, no RGB->Y
                    psnr = cpsnr(predict_y, ground_truth)
                    ssim = cssim(predict_y, ground_truth, 255)
                    sam = csam(predict_y, ground_truth) * 180.0 / np.pi
                    ergas = cergas(predict_y, ground_truth, scale=scale)
                    batch_psnr.append(psnr)
                    batch_ssim.append(ssim)
                    batch_sam.append(sam)
                    batch_ergas.append(ergas)
                avg_psnr = np.array(batch_psnr).mean()
                avg_ssim = np.array(batch_ssim).mean()
                avg_sam = np.array(batch_sam).mean()
                avg_ergas = np.array(batch_ergas).mean()
                psnr_list.extend(batch_psnr)
                ssim_list.extend(batch_ssim)
                sam_list.extend(batch_sam)
                ergas_list.extend(batch_ergas)
                t1.set_postfix_str('n:Batch loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}, SAM: {:.4f}, ERGAS: {:.4f}'.format(
                    loss.item(), avg_psnr, avg_ssim, avg_sam, avg_ergas))
                t1.update()
            if 'SAM' not in self.records:
                self.records['SAM'] = []
                self.records['ERGAS'] = []
            self.records['Epoch'].append(self.epoch)
            self.records['PSNR'].append(np.array(psnr_list).mean())
            self.records['SSIM'].append(np.array(ssim_list).mean())
            self.records['SAM'].append(np.array(sam_list).mean())
            self.records['ERGAS'].append(np.array(ergas_list).mean())

            save_config(self.log_name, 'Val Epoch {}: PSNR={:.4f}, SSIM={:.6f}, SAM={:.4f}, ERGAS={:.4f}'.format(
                self.epoch, self.records['PSNR'][-1], self.records['SSIM'][-1],
                self.records['SAM'][-1], self.records['ERGAS'][-1]))
            self.writer.add_scalar('PSNR_epoch', self.records['PSNR'][-1], self.epoch)
            self.writer.add_scalar('SSIM_epoch', self.records['SSIM'][-1], self.epoch)
            self._swan_log({
                'val/psnr': self.records['PSNR'][-1],
                'val/ssim': self.records['SSIM'][-1],
                'val/sam': self.records['SAM'][-1],
                'val/ergas': self.records['ERGAS'][-1],
            })

    def _forward(self, lms_image, bms_image, pan_image):
        out = self.model(lms_image, bms_image, pan_image)
        return unwrap_hrms(out), out if isinstance(out, dict) else None

    def check_gpu(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            cudnn.benchmark = True
              
            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >=0:
                    self.gpu_ids.append(gid)

            torch.cuda.set_device(self.gpu_ids[0]) 
            self.loss = self.loss.cuda(self.gpu_ids[0])
            #self.vggloss = self.vggloss.cuda(self.gpu_ids[0])
            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids) 

    def check_pretrained(self):
        checkpoint = os.path.join(self.cfg['pretrain']['pre_folder'], self.cfg['pretrain']['pre_sr'])
        if os.path.exists(checkpoint):
            ckpt = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(ckpt['net'])
            if ckpt.get('optimizer') is not None:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                except Exception:
                    pass
            self.epoch = int(ckpt.get('epoch', 0)) + 1
            self.records = {'Epoch': [], 'PSNR': [], 'SSIM': [], 'Loss': [], 'SAM': [], 'ERGAS': []}
            if self.epoch > self.nEpochs:
                raise Exception("Pretrain epoch must less than the max epoch!")
        else:
            raise Exception("Pretrain path error!")

    def save_checkpoint(self):
        super(Solver, self).save_checkpoint()
        self.ckp['net'] = self.model.state_dict()
        self.ckp['optimizer'] = self.optimizer.state_dict()
        if not os.path.exists(self.cfg['checkpoint'] + '/' + str(self.log_name)):
            os.mkdir(self.cfg['checkpoint'] + '/' + str(self.log_name))
        torch.save(self.ckp, os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'))

        if self.cfg['save_best']:
            if self.records['SSIM'] != [] and self.records['SSIM'][-1] == np.array(self.records['SSIM']).max():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'bestSSIM.pth'))
            if self.records['PSNR'] !=[] and self.records['PSNR'][-1]==np.array(self.records['PSNR']).max():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'bestPSNR.pth'))

    def run(self):
        self.check_gpu()
        if self.cfg['pretrain']['pretrained']:
            self.check_pretrained()
        try:
            while self.epoch <= self.nEpochs:
                self.train()
                self.eval()
                self.save_checkpoint()
                self.epoch += 1
        except KeyboardInterrupt:
            self.save_checkpoint()
        save_config(self.log_name, 'Training done.')
        self._swan_finish()

    def _swanlab_api_key(self):
        key = os.environ.get('SWANLAB_API_KEY')
        if key:
            return key.strip()
        key_file = os.environ.get('SWANLAB_API_KEY_FILE', '')
        if os.path.isfile(key_file):
            with open(key_file, 'r') as f:
                return f.read().strip()
        return None

    def _init_swanlab(self):
        self.swan = None
        sl = self.cfg.get('swanlab') if isinstance(self.cfg.get('swanlab'), dict) else {}
        if not sl.get('enable', False):
            return
        try:
            import swanlab
        except ImportError:
            print('swanlab not installed, skip cloud logging')
            return
        api_key = self._swanlab_api_key()
        if api_key:
            swanlab.login(api_key=api_key, save=False)
        train_dir = self.cfg.get('data_dir_train', '')
        dataset = os.path.basename(os.path.dirname(train_dir.rstrip('/\\'))) or 'unknown'
        experiment = sl.get('experiment') or '{}-{}-{}'.format(
            self.cfg.get('algorithm'), dataset, self.timestamp
        )
        config = {
            'algorithm': self.cfg.get('algorithm'),
            'name': self.cfg.get('name'),
            'dataset': dataset,
            'data_dir_train': train_dir,
            'epochs': self.cfg.get('nEpochs'),
            'batch_size': self.cfg.get('data', {}).get('batch_size'),
            'lr': self.cfg.get('schedule', {}).get('lr'),
            'scale': self.cfg.get('data', {}).get('upsacle'),
            'log_name': self.log_name,
        }
        mq = self.cfg.get('mq') if isinstance(self.cfg.get('mq'), dict) else {}
        config.update(dict(('mq_' + k, v) for k, v in mq.items()))
        self.swan = swanlab.init(
            project=sl.get('project', 'pansharpening'),
            experiment_name=experiment,
            description=sl.get('description', ''),
            config=config,
            logdir=os.path.join(self.cfg.get('log_dir', './log/'), 'swanlog'),
        )
        save_config(self.log_name, 'SwanLab experiment: ' + experiment)

    def _swan_log(self, metrics):
        if self.swan is None:
            return
        import swanlab
        data = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().item()
            data[key] = float(value)
        swanlab.log(data, step=int(self.epoch))

    def _swan_finish(self):
        if self.swan is None:
            return
        import swanlab
        swanlab.finish()