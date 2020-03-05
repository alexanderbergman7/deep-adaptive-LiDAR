import os
import argparse
import torch
import time
from tensorboardX import SummaryWriter
import torch.nn.functional as F

# from models.models_refinement import Refinement
from models.models_std import DepthCompletionNet as Refinement
from models.densedepth import MonocularDepth
from models.pytorch_prototyping_orig import Unet
from dataloaders.kitti import GetKittiData
# from dataloaders.kitti_inpainted import GetKittiData
# from dataloaders.nyu_v2 import GetNYUV2Data
from dataloaders.nyu_v2_wonka import GetNYUV2Data

import dataloaders.transforms_kitti as transforms_kitti
import dataloaders.transforms_nyu as transforms_nyu
import torchvision.transforms
import utils
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description='Monocular Depth Refinement')
parser.add_argument('--epochs',
                    default=16,
                    type=int,
                    help='number of total epochs to run for')
parser.add_argument('-bs',
                    '--batch_size',
                    default=16,
                    type=int,
                    help='batch size')
parser.add_argument('--lr',
                    default=1e-3,
                    type=float,
                    help='initial learning rate')
parser.add_argument('--lr_decay',
                    default=0.8,
                    type=float,
                    help='learning rate decay steps (every 2 epochs)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    help='path to resume from a saved model')
parser.add_argument('--sampling_method',
                    default='u_r',
                    type=str,
                    choices=['u_r', 'grid', 'pdisk', 'segmentation', 'oracle', 'none'],
                    help='method for generating the sparse samples')
parser.add_argument('--name',
                    default='',
                    type=str,
                    help='extra strings to add to name of model dir')
parser.add_argument('-s',
                    '--samples',
                    default=156,
                    type=int,
                    help='number of sparse samples')
parser.add_argument('-gpu',
                    type=int,
                    help='gpu to run on')
parser.add_argument('--dataset',
                    type=str,
                    default='kitti',
                    choices=['kitti', 'nyu_v2'],
                    help='dataset to train on')
parser.add_argument('--save_directory',
                    type=str,
                    default='/media/data2/awb/adaptive-scanning/'
                    + 'results_refinement_final',
                    help='directory to save files in')
parser.add_argument('--eval_only',
                    type=str,
                    default=False,
                    help='evaluate only')

# existing model parameters
parser.add_argument('--monocular_depth_model_nyu',
                    type=str,
                    default='/home/awb/adaptive-scanning/model/densedepth/'
                       + 'densedepth_ledges_third_nonshuffled_17.pt',
                    help='path to monocular depth estimation model')
parser.add_argument('--bilat_proxy_model_nyu',
                    type=str,
                    default='/home/awb/adaptive-scanning/model/bilateral_proxy/'
                       + '50_fullnyuv2_moreLRmodern_best.pth.tar',
                    help='path to bilateral filter proxy')
parser.add_argument('--monocular_depth_model_kitti',
                    type=str,
                    default='/home/awb/adaptive-scanning/model/densedepth/'
                       + 'densedepth_KITTI_first_third_17.pt',
                    help='path to monocular depth estimation model')
parser.add_argument('--bilat_proxy_model_kitti',
                    type=str,
                    default='/home/awb/adaptive-scanning/model/bilateral_proxy/'
                       + '512_fullkitti_depthdomain_best.pth.tar',
                    help='path to bilateral filter proxy')
parser.add_argument('--data_directory_nyu',
                    type=str,
                    default='/media/data2/awb/nyu_v2/densedepth/nyu_data_wonka.zip',
                    help='directory containing the kitti data')
parser.add_argument('--data_directory_kitti',
                    type=str,
                    default='/media/data2/awb/kitti/' \
                            'sparse-to-dense/self-supervised-depth-completion/data',
                    help='directory containing the nyu v2 data')
args = parser.parse_args()

if args.sampling_method == "none":
    args.samples = 0
current_time = time.strftime('%Y-%m-%d@%H-%M')
args.save_directory = os.path.join(args.save_directory,
                    "data={}.samples={}.method={}.bs={}.lr={}.time={}".
                    format(args.dataset, args.samples,
                    args.sampling_method, args.batch_size,
                    args.lr, current_time))
if args.name:
    args.save_directory = args.save_directory + '_' + args.name
writer = SummaryWriter(os.path.join(args.save_directory, "logfile"))

print(args)
device = torch.device("cuda:"+str(args.gpu))


def validate(val_loader, models, logr, epoch):
    model, bprox, mdn = models
    model = model.eval()
    block_average_meter = utils.AverageMeter()
    average_meter = utils.AverageMeter()

    for i, batch_data in enumerate(val_loader):
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }

        # monocular depth estimation in range of (2,1000) as in paper
        # scale to be in depth range of the appropriate dataset
        # bilateral proxy was also trained to take images in range of (2,1000)
        # and output images in range of (2, 1000)
        if args.dataset == 'kitti':
            mdi = torch.clamp(F.interpolate(utils.DepthNorm(
                mdn(batch_data['rgb'])), scale_factor=2), 2, 1000) / 1000 * 85
            inpainted = bprox(torch.cat([mdi / 85 * 1000,
                                         batch_data['d'] / 85 * 1000], dim=1)
                              ) / 1000 * 85
        elif args.dataset == 'nyu_v2':
            mdi = torch.clamp(utils.DepthNorm(mdn(batch_data['rgb'])),
                              10, 1000) / 1000 * 10
            inpainted = bprox(torch.cat([mdi / 10 * 1000,
                                         batch_data['d'] / 10 * 1000], dim=1)
                              ) / 1000 * 10
        else:
            print("invalid dataset")
            exit()

        # add bilateral proxy and monocular depth estimate to the data to feed
        # to the inpainting model
        batch_data['bproxi'] = inpainted
        batch_data['mdi'] = mdi

        pred = model(batch_data)

        result = utils.Result()

        # uncomment line below if using the inpainted dataset and want to train with
        # the sparse KITTI data instead of inpainted kitti data
        # result.evaluate(pred, batch_data['gt_sparse'])
        result.evaluate(pred, batch_data['gt'])

        # result.evaluate(crop(pred, 228, 304), crop(batch_data['gt'], 228, 304))
        block_average_meter.update(result)
        average_meter.update(result)
        if (i+1) % 20 == 0:
            logr.print(i, epoch, args.lr, len(val_loader), block_average_meter,
                       average_meter)

        logr.conditional_save_img_comparison(i, batch_data, pred, epoch, args.dataset)

    avg = average_meter.average()
    is_best = logr.rank_save_best(avg, epoch)
    logr.summarize(avg, is_best)

    return avg, is_best

def train_epoch(train_loader, models, optimizer, logr, epoch):
    model, bprox, mdn = models
    model = model.train()
    block_average_meter = utils.AverageMeter()
    average_meter = utils.AverageMeter()

    for i, batch_data in enumerate(train_loader):
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }

        # monocular depth estimation in range of (2,1000) as in paper
        # scale to be in depth range of the appropriate dataset
        # bilateral proxy was also trained to take images in range of (2,1000)
        # and output images in range of (2, 1000)
        if args.dataset == 'kitti':
            mdi = torch.clamp(F.interpolate(utils.DepthNorm(
                mdn(batch_data['rgb'])), scale_factor=2), 2, 1000) / 1000 * 85
            inpainted = bprox(torch.cat([mdi / 85 * 1000,
                                         batch_data['d'] / 85 * 1000], dim=1)
                              ) / 1000 * 85
        elif args.dataset == 'nyu_v2':
            mdi = torch.clamp(utils.DepthNorm(mdn(batch_data['rgb'])),
                              10, 1000) / 1000 * 10
            inpainted = bprox(torch.cat([mdi / 10 * 1000,
                                         batch_data['d'] / 10 * 1000], dim=1)
                              ) / 1000 * 10
        else:
            print("invalid dataset")
            exit()

        batch_data['bproxi'] = inpainted
        batch_data['mdi'] = mdi

        pred = model(batch_data)
        loss = utils.loss(pred, batch_data, args.dataset)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('data/train_loss', loss.cpu().data.numpy(),
                               epoch*len(train_loader) + i)
        result = utils.Result()
        result.evaluate(pred, batch_data['gt'])
        block_average_meter.update(result)
        average_meter.update(result)
        if (i+1) % 20 == 0:
            logr.print(i, epoch, args.lr, len(train_loader), block_average_meter,
                       average_meter)


def main():
    global args
    checkpoint = None
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("loading checkpoint '{}'".format(args.resume), end='')
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
        else:
            print("no checkpoint found at '{}".format(args.resume), end='')
            return

    print("creating model and optimizer")
    model = Refinement()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2,
                                                    gamma=args.lr_decay,
                                                    last_epoch=checkpoint['epoch'])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2,
                                                    gamma=args.lr_decay)

    print("loading monocular depth and bilateral proxy models")
    if args.dataset == 'kitti':
        bprox = Unet(in_channels=2, out_channels=1, nf0=4, num_down=4,
                     max_channels=512, use_dropout=False)
        bprox.load_state_dict(
            torch.load(args.bilat_proxy_model_kitti,
                       map_location=device)['network'])
        bprox.to(device)
        bprox.eval()
        mdn = MonocularDepth()
        mdn.load_state_dict(
            torch.load(args.monocular_depth_model_kitti,
                       map_location=device))
        mdn.to(device)
        mdn.eval()
    elif args.dataset =='nyu_v2':
        bprox = Unet(in_channels=2, out_channels=1, nf0=4, num_down=4,
                     max_channels=512, use_dropout=False)
        bprox.load_state_dict(
            torch.load(args.bilat_proxy_model_nyu,
                       map_location=device)['network'])
        bprox.to(device)
        bprox.eval()
        mdn = MonocularDepth()
        mdn.load_state_dict(
            torch.load(args.monocular_depth_model_nyu,
                       map_location=device))
        mdn.to(device)
        mdn.eval()
    else:
        print("not a valid dataset")
        return

    model.to(device)

    # uncomment the below line for multiple GPU processing, have to hard-code the
    # additional GPU IDs:
    # model = torch.nn.DataParallel(model, device_ids=[args.gpu, 9])
    # bprox = torch.nn.DataParallel(bprox, device_ids=[args.gpu, 9])
    # mdn = torch.nn.DataParallel(mdn, device_ids=[args.gpu, 9])
    models = (model, bprox, mdn)

    print("creating data loaders")
    if args.dataset == 'kitti':
        train_loader, val_loader = GetKittiData(batch_size=args.batch_size,
                                                samples=args.samples,
                                                sampling_method=
                                                args.sampling_method,
                                                data_directory=args.data_directory)
        print("train loader size: {}".format(len(train_loader)))
        print("val loader size: {}".format(len(val_loader)))
    elif args.dataset == 'nyu_v2':
        train_loader, val_loader = GetNYUV2Data(batch_size=args.batch_size,
                                                samples=args.samples,
                                                sampling_method=
                                                args.sampling_method,
                                                data_directory=args.data_directory)
        print("train loader size: {}".format(len(train_loader)))
        print("val loader size: {}".format(len(val_loader)))
    else:
        print("not a valid dataset")
        return

    logr = utils.logger(args)
    if checkpoint is not None:
        logr.best_result = checkpoint['best_result']
    print("logger created")

    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=device)['model'])
        model.to(device)
        validate(val_loader, models, logr, 0)
        return

    for epoch in range(start_epoch, args.epochs):
        print("beginning epoch: {}".format(epoch))
        train_epoch(train_loader, models, optimizer, logr, epoch)
        result, is_best = validate(val_loader, models, logr, epoch)

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'best_result': logr.best_result,
            'optimizer': optimizer.state_dict(),
            'args': args
        }, os.path.join(args.save_directory, 'checkpoint.pth.tar'))
        if is_best:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'best_result': logr.best_result,
                'optimizer': optimizer.state_dict(),
                'args': args
            }, os.path.join(args.save_directory, 'model_best.pth.tar'))

        scheduler.step()


if __name__ == '__main__':
    main()
