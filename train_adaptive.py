import os
import argparse
import torch
import time
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from models.models_refinement import Refinement
from models.models_std import DepthCompletionNet
from models.densedepth import MonocularDepth
from models.pytorch_prototyping_orig import Unet
from models.parameter_prediction import SparseDepthPrediction
from dataloaders.kitti_inpainted import GetKittiData
# from dataloaders.nyu_v2 import GetNYUV2Data
from dataloaders.nyu_v2_wonka import GetNYUV2Data
import metrics

import utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description='Adaptive Sampling')
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
                    default=1e-5,
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
parser.add_argument('--name',
                    default='',
                    type=str,
                    help='extra strings to add to name of model dir')
parser.add_argument('-gpu',
                    type=int,
                    default=0,
                    help='gpu to run on')
parser.add_argument('--dataset',
                    type=str,
                    default='kitti',
                    choices=['kitti', 'nyu_v2'],
                    help='dataset to train on')
parser.add_argument('--save_directory',
                    type=str,
                    default='/media/data2/awb/adaptive-scanning/'
                    + 'results_adaptive_final',
                    help='directory to save files in')
parser.add_argument('-sH',
                    '--samples_height',
                    default=6,
                    type=int,
                    help='number of sparse samples in grid along height dim')
parser.add_argument('-sW',
                    '--samples_width',
                    default=8,
                    type=int,
                    help='number of sparse samples in grid along width dim')
parser.add_argument('--end-to-end',
                    action='store_true',
                    help='train end-to-end samples and network')
parser.add_argument('--grid_reg',
                    type=float,
                    default=5e3,
                    help='Regularization to stay on the grid')
parser.add_argument('--image_reg',
                    type=float,
                    default=1e10,
                    help='Regularization to keep samples within the image')
parser.add_argument('--eval_only',
                    type=str,
                    default=False,
                    help='evaluate only')
parser.add_argument('--ret_samples',
                    action='store_true',
                    help='return the number of samples in the image')

parser.add_argument('--adaptive_sampling_only_train',
                    action='store_true',
                    help='train only the adaptive sampling model, not end-to-end')
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
parser.add_argument('--refinement_model_kitti',
                    type=str,
                    default='/media/data2/awb/adaptive-scanning/'
                       + 'results_awb/kitti_models/'
                       # + 'model_best_156.pth.tar',
                       + 'model_best_512.pth.tar',
                       # + 'model_best_156_130prox.pth.tar',
                    help='path to refinement model on kitti data')
parser.add_argument('--refinement_model_nyu',
                    type=str,
                    default='/media/data2/awb/adaptive-scanning/'
                       + 'results_refinement_final/final_models/'
                       + 'data-nyu_v2.samples-50.pth.tar',
                    help='path to refinement model on nyu data')
parser.add_argument('--data_directory_nyu',
                    type=str,
                    default='/media/data2/awb/nyu_v2/densedepth/nyu_data_wonka.zip',
                    help='directory containing the data')
parser.add_argument('--data_directory_kitti',
                    type=str,
                    default='/media/data2/awb/kitti/inpainted',
                    help='directory containing the data')
args = parser.parse_args()

current_time = time.strftime('%Y-%m-%d@%H-%M')
args.save_directory = os.path.join(args.save_directory,
                    "time={}.data={}.samples={}-{}.bs={}.lr={}.greg={}.ireg={}".
                    format(current_time, args.dataset, args.samples_height,
                    args.samples_width, args.batch_size,
                    args.lr, args.grid_reg, args.image_reg))
if args.name:
    args.save_directory = args.save_directory + '_' + args.name

writer = SummaryWriter(os.path.join(args.save_directory, "logfile"))

print(args)
device = torch.device("cuda:"+str(args.gpu))

def validate(val_loader, models, logr, epoch, args, device):
    as_model, bprox, mdn, depthcomp = models
    as_model.eval()
    block_average_meter = utils.AverageMeter()
    average_meter = utils.AverageMeter()

    if args.ret_samples:
        samps_tot = 0

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
            with torch.no_grad():
                mdi = torch.clamp(F.interpolate(utils.DepthNorm(
                    mdn(batch_data['rgb'])), scale_factor=2), 2, 1000) / 1000 * 85
        elif args.dataset == 'nyu_v2':
            with torch.no_grad():
                mdi = torch.clamp(utils.DepthNorm(mdn(batch_data['rgb'])),
                                  10, 1000) / 1000 * 10
        else:
            print("invalid dataset")
            exit()

        batch_data['mdi'] = mdi

        if args.ret_samples:
            pred_sparse_depth, vector_field, grid_to_sample, samps = as_model(batch_data)
            samps_tot += samps
        else:
            pred_sparse_depth, vector_field, grid_to_sample = as_model(batch_data)

        batch_data['d'] = pred_sparse_depth

        # monocular depth estimation in range of (2,1000) as in paper
        # scale to be in depth range of the appropriate dataset
        # bilateral proxy was also trained to take images in range of (2,1000)
        # and output images in range of (2, 1000)
        if args.dataset == 'kitti':
            inpainted = bprox(torch.cat([mdi / 85 * 1000,
                                         pred_sparse_depth / 85 * 1000], dim=1)
                              ) / 1000 * 85
        elif args.dataset == 'nyu_v2':
            inpainted = bprox(torch.cat([mdi / 10 * 1000,
                                         pred_sparse_depth / 10 * 1000], dim=1)
                              ) / 1000 * 10
        else:
            print("invalid dataset")
            exit()

        batch_data['bproxi'] = inpainted
        pred = depthcomp(batch_data)

        # pause at some iteration to save a full pipeline figure for a validation image
        # if i == 341:
        #     utils.save_methods_figure(batch_data, pred_sparse_depth, vector_field, pred)
        #     exit()

        # evaluate on the sparse ground truth datapoints
        result = utils.Result()
        if args.dataset == 'nyu_v2':
            result.evaluate(pred, batch_data['gt'])
        elif args.dataset == 'kitti':
            result.evaluate(pred, batch_data['gt_sparse'])
        block_average_meter.update(result)
        average_meter.update(result)
        if (i+1) % 200 == 0:
            logr.print(i, epoch, args.lr, len(val_loader), block_average_meter,
                       average_meter)

        logr.conditional_save_img_comparison(i, batch_data, pred, epoch, args.dataset)

    avg = average_meter.average()
    is_best = logr.rank_save_best(avg, epoch)
    logr.summarize(avg, is_best)

    writer.add_scalar('data/val_loss_rmse', avg.rmse,
                      epoch)

    if args.ret_samples:
        print("AVERAGE SAMPLES:")
        print(samps_tot / len(val_loader))

    return avg, is_best


def train_epoch(train_loader, models, optimizer, logr, epoch, args, device):
    as_model, bprox, mdn, depthcomp = models
    as_model.train()
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
        elif args.dataset == 'nyu_v2':
            with torch.no_grad():
                mdi = torch.clamp(utils.DepthNorm(mdn(batch_data['rgb'])),
                                  10, 1000) / 1000 * 10
        else:
            print("invalid dataset")
            exit()

        batch_data['mdi'] = mdi

        # predict sampling locations
        pred_sparse_depth, vector_field, grid_to_sample = as_model(batch_data)
        batch_data['d'] = pred_sparse_depth

        # monocular depth estimation in range of (2,1000) as in paper
        # scale to be in depth range of the appropriate dataset
        # bilateral proxy was also trained to take images in range of (2,1000)
        # and output images in range of (2, 1000)
        if args.dataset == 'kitti':
            inpainted = bprox(torch.cat([mdi / 85 * 1000,
                                         pred_sparse_depth / 85 * 1000], dim=1)
                              ) / 1000 * 85
        elif args.dataset == 'nyu_v2':
            inpainted = bprox(torch.cat([mdi / 10 * 1000,
                                         pred_sparse_depth / 10 * 1000], dim=1)
                              ) / 1000 * 10
        else:
            print("invalid dataset")
            exit()

        batch_data['bproxi'] = inpainted
        pred = depthcomp(batch_data)

        loss = utils.adaptive_loss(pred, vector_field, grid_to_sample,
                                   batch_data, args.dataset,
                                   args.grid_reg, args.image_reg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('data/train_loss_full', loss.cpu().data.numpy(),
                          epoch*len(train_loader) + i)
        result = utils.Result()
        result.evaluate(pred, batch_data['gt'])
        block_average_meter.update(result)
        average_meter.update(result)
        if (i+1) % 20 == 0:
            logr.print(i, epoch, args.lr, len(train_loader), block_average_meter,
                       average_meter)

        writer.add_scalar('data/train_loss_rmse', result.rmse,
                          epoch * len(train_loader) + i)


def main(args, device):
    # global args
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
    if args.dataset == 'kitti':
        as_model = SparseDepthPrediction(args.samples_height, args.samples_width,
                                      (args.batch_size, 1, 352, 1216),
                                      device=device, dataset=args.dataset, ret_samples=args.ret_samples)
    elif args.dataset == 'nyu_v2':
        as_model = SparseDepthPrediction(args.samples_height, args.samples_width,
                                      (args.batch_size, 1, 240, 320),
                                      device=device, dataset=args.dataset, ret_samples=args.ret_samples)

    if args.adaptive_sampling_only_train:
        optimizer = torch.optim.Adam(as_model.parameters(), lr=args.lr)

        if checkpoint is not None:
            as_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2,
                                                        gamma=args.lr_decay,
                                                        last_epoch=checkpoint['epoch'])

            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        else:
            pass
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2,
                                                        gamma=args.lr_decay)

    print("loading monocular depth, bilateral proxy, and refinement models")
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
        depthcomp = DepthCompletionNet()
        depthcomp.load_state_dict(
            torch.load(args.refinement_model_kitti,
                       map_location=device)['model'])
        depthcomp.to(device)
        depthcomp.eval()
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
        depthcomp = Refinement()
        depthcomp.load_state_dict(
            torch.load(args.refinement_model_nyu,
                       map_location=device)['model'])

        depthcomp.to(device)
        depthcomp.eval()
    else:
        print("not a valid dataset")
        return

    as_model.to(device)

    if not args.adaptive_sampling_only_train:
        optimizer = torch.optim.Adam(list(as_model.parameters()) +
                                     list(depthcomp.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2,
                                                    gamma=args.lr_decay)

    # uncomment for multiple GPU training, manually specify remaining GPUs
    # model = torch.nn.DataParallel(model, device_ids=[args.gpu, 9])
    # bprox = torch.nn.DataParallel(bprox, device_ids=[args.gpu, 9])
    # mdn = torch.nn.DataParallel(mdn, device_ids=[args.gpu, 9])
    models = (as_model, bprox, mdn, depthcomp)

    print("creating data loaders")
    if args.dataset == 'kitti':
        train_loader, val_loader = GetKittiData(batch_size=args.batch_size,
                                                samples=0,
                                                sampling_method='u_r',
                                                data_directory=args.data_directory_kitti)
        print("train loader size: {}".format(len(train_loader)))
        print("val loader size: {}".format(len(val_loader)))
    elif args.dataset == 'nyu_v2':
        train_loader, val_loader = GetNYUV2Data(batch_size=args.batch_size,
                                                samples=0,
                                                sampling_method='u_r',
                                                data_directory=args.data_directory_nyu)
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
        as_model.load_state_dict(torch.load(args.eval_only, map_location=device)['model'])
        as_model.to(device)
        validate(val_loader, models, logr, 0, args, device)
        return

    for epoch in range(start_epoch, args.epochs):
        print("beginning epoch: {}".format(epoch))
        train_epoch(train_loader, models, optimizer, logr, epoch, args, device)
        result, is_best = validate(val_loader, models, logr, epoch, args, device)

        torch.save({
            'epoch': epoch,
            'model': as_model.state_dict(),
            'dc_model': depthcomp.state_dict(),
            'best_result': logr.best_result,
            'optimizer': optimizer.state_dict(),
            'args': args
        }, os.path.join(args.save_directory, 'checkpoint.pth.tar'))
        if is_best:
            torch.save({
                'epoch': epoch,
                'model': as_model.state_dict(),
                'dc_model': depthcomp.state_dict(),
                'best_result': logr.best_result,
                'optimizer': optimizer.state_dict(),
                'args': args
            }, os.path.join(args.save_directory, 'model_best.pth.tar'))

        scheduler.step()


if __name__ == '__main__':
    main(args, device)
