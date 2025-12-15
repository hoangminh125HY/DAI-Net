# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from torch.utils.data import DataLoader
import os
import random
import time
import torch
import argparse
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

from data.config import cfg
from layers.modules import MultiBoxLoss, EnhanceLoss
from data.widerface import WIDERDetection, detection_collate
from models.factory import build_net, basenet_factory
from models.enhancer import RetinexNet
from utils.DarkISP import Low_Illumination_Degrading
from PIL import Image

parser = argparse.ArgumentParser(
    description='DSFD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size',
                    default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--model',
                    default='dark', type=str,
                    choices=['dark', 'vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    action='store_true',
                    default=True,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=5e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=True, type=bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--local_rank',
                    type=int,
                    help='local rank for dist')

args = parser.parse_args()
global local_rank
local_rank = args.local_rank

if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)

if torch.cuda.is_available():
    if args.cuda:
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        import torch.distributed as dist

        gpu_num = torch.cuda.device_count()
        if local_rank == 0:
            print('Using {} gpus'.format(gpu_num))
        rank = int(os.environ['RANK'])
        torch.cuda.set_device(rank % gpu_num)
        dist.init_process_group('nccl')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

save_folder = os.path.join(args.save_folder, args.model)
if not os.path.exists(save_folder):
    os.makedirs(save_folder, exist_ok=True)


train_dataset = WIDERDetection('/kaggle/input/data-2017/dts_2017/my_night/annotations/train2017.json', mode='train')
val_dataset = WIDERDetection('/kaggle/input/data-2017/dts_2017/my_night/annotations/val2017.json', mode='val')

# Sử dụng DataLoader bình thường
train_loader = DataLoader(train_dataset, args.batch_size, 
                          num_workers=args.num_workers, 
                          collate_fn=detection_collate, 
                          pin_memory=True)

val_loader = DataLoader(val_dataset, args.batch_size, 
                        num_workers=0, 
                        collate_fn=detection_collate, 
                        pin_memory=True)

min_loss = np.inf


def train():
    # Kiểm tra số lượng GPU có sẵn
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("No GPU available. Training on CPU.")
        per_epoch_size = len(train_dataset) // args.batch_size  # Dùng CPU
    else:
        per_epoch_size = len(train_dataset) // (args.batch_size * gpu_count)  # Dùng GPU

    start_epoch = 0
    iteration = 0
    step_index = 0

    # Khởi tạo mô hình
    basenet = basenet_factory(args.model)
    dsfd_net = build_net('train', cfg.NUM_CLASSES, args.model)
    net = dsfd_net
    net_enh = RetinexNet()
    model_path = os.path.join(args.save_folder, 'decomp.pth')

    # Kiểm tra sự tồn tại của tệp decomp.pth
    if os.path.exists(model_path):
        net_enh.load_state_dict(torch.load(model_path))
    else:
        print(f"Warning: {model_path} not found. Initializing model from scratch.")
        net_enh.apply(net_enh.weights_init)

    # Tiến hành huấn luyện nếu có checkpoint
    if args.resume:
        print(f'Resuming training, loading {args.resume}...')
        start_epoch = net.load_weights(args.resume)
        iteration = start_epoch * per_epoch_size
    else:
        base_weights = torch.load(args.save_folder + basenet)
        print(f'Load base network {args.save_folder + basenet}')
        if args.model == 'vgg' or args.model == 'dark':
            net.vgg.load_state_dict(base_weights)
        else:
            net.resnet.load_state_dict(base_weights)

    # Khởi tạo weights nếu không có checkpoint
    if not args.resume:
        print('Initializing weights...')
        net.extras.apply(net.weights_init)
        net.fpn_topdown.apply(net.weights_init)
        net.fpn_latlayer.apply(net.weights_init)
        net.fpn_fem.apply(net.weights_init)
        net.loc_pal1.apply(net.weights_init)
        net.conf_pal1.apply(net.weights_init)
        net.loc_pal2.apply(net.weights_init)
        net.conf_pal2.apply(net.weights_init)
        net.ref.apply(net.weights_init)

    # Scaling the learning rate
    lr = args.lr * np.round(np.sqrt(args.batch_size / 4 * gpu_count), 4)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.cuda and gpu_count > 0:
        net = net.cuda()  # Chuyển mô hình lên GPU nếu có GPU

    criterion = MultiBoxLoss(cfg, args.cuda)
    criterion_enhance = EnhanceLoss()

    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        net.train()

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images = Variable(images.cuda() / 255.) if args.cuda else Variable(images / 255.)
            targetss = [Variable(ann.cuda(), requires_grad=False) for ann in targets] if args.cuda else [Variable(ann, requires_grad=False) for ann in targets]

            # Tạo ảnh bị giảm sáng (degraded)
            img_dark = torch.empty(size=(images.shape[0], images.shape[1], images.shape[2], images.shape[3])).cuda() if args.cuda else torch.empty(size=(images.shape[0], images.shape[1], images.shape[2], images.shape[3]))

            for i in range(images.shape[0]):
                img_dark[i], _ = Low_Illumination_Degrading(images[i])

            t0 = time.time()
            R_dark_gt, I_dark = net_enh(img_dark)
            R_light_gt, I_light = net_enh(images)

            out, out2, loss_mutual = net(img_dark, images, I_dark.detach(), I_light.detach())
            R_dark, R_light, R_dark_2, R_light_2 = out2

            optimizer.zero_grad()

            loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targetss)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targetss)

            loss_enhance = criterion_enhance([R_dark, R_light, R_dark_2, R_light_2, I_dark.detach(), I_light.detach()], images, img_dark) * 0.1
            loss_enhance2 = F.l1_loss(R_dark, R_dark_gt.detach()) + F.l1_loss(R_light, R_light_gt.detach()) + (
                        1. - ssim(R_dark, R_dark_gt.detach())) + (1. - ssim(R_light, R_light_gt.detach()))

            loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2 + loss_enhance2 + loss_enhance + loss_mutual
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=35, norm_type=2)
            optimizer.step()

            t1 = time.time()
            losses += loss.item()

            if iteration % 100 == 0:
                tloss = losses / (batch_idx + 1)
                print(f"Timer: {t1 - t0:.4f}")
                print(f"epoch:{epoch} || iter:{iteration} || Loss:{tloss:.4f}")
                print(f"->> pal1 conf loss:{loss_c_pal1.item():.4f} || pal1 loc loss:{loss_l_pa1l.item():.4f}")
                print(f"->> pal2 conf loss:{loss_c_pal2.item():.4f} || pal2 loc loss:{loss_l_pa12.item():.4f}")
                print(f"->>lr:{optimizer.param_groups[0]['lr']}")

            if iteration % 5000 == 0:
                print(f'Saving state, iter: {iteration}')
                torch.save(net.state_dict(), os.path.join(save_folder, f'dsfd_{iteration}.pth'))

            iteration += 1

        if epoch >= 0:
            val(epoch, net, criterion)

        if iteration >= cfg.MAX_STEPS:
            break


def val(epoch, net, criterion):
    net.eval()
    losses = 0
    t1 = time.time()

    for batch_idx, (images, targets, img_paths) in enumerate(val_loader):
        images = Variable(images.cuda() / 255.) if args.cuda else Variable(images / 255.)
        targets = [Variable(ann.cuda(), volatile=True) for ann in targets] if args.cuda else [Variable(ann, volatile=True) for ann in targets]

        img_dark = torch.stack([Low_Illumination_Degrading(images[i])[0] for i in range(images.shape[0])], dim=0)

        out, R = net(img_dark)

        loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
        loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
        loss = loss_l_pa12 + loss_c_pal2

        losses += loss.item()

    print(f'Validation Loss after epoch {epoch}: {losses / len(val_loader)}')


if __name__ == '__main__':
    train()