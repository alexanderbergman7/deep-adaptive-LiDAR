import os
if not ("DISPLAY" in os.environ):
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
cmap = plt.cm.jet

import torch
import math
from math import exp
import torch.nn.functional as F
import torch.nn as nn
import scipy.io as sio

from math import cos, sin, floor, sqrt, pi, ceil
from random import random

MSE = nn.MSELoss()

def MaskedMSELoss(pred, target):
    assert pred.dim() == target.dim(), "inconsistent dimensions"
    valid_mask = (target > 0).detach()
    diff = target - pred
    diff = diff[valid_mask]
    return (diff**2).mean()


def DepthNorm(depth, maxDepth=1000.0):
    return maxDepth / depth


def loss(pred, batch_data, data):
    """Loss function with MSE and SSIM for training refinement"""
    target = batch_data['gt']
    mdi = batch_data['mdi']
    if data == 'kitti':
        return MaskedMSELoss(pred, target) + 0.5 * \
            torch.clamp((1 - ssim(pred, mdi, val_range=85.0)) * 0.5, 0, 1)
    elif data == 'nyu_v2':
        return MaskedMSELoss(pred, target) + 0.5 * \
            torch.clamp((1 - ssim(pred, mdi, val_range=10.0)) * 0.5, 0, 1)
    else:
        print("invalid data")
        exit()

def adaptive_loss(pred, vector_field, grid_to_sample,
                  batch_data, dataset, grid_reg, image_reg):
    """Loss function for training end-to-end: MSE loss (can include SSIM if desired), grid loss
    as described in paper, image loss as described in the paper"""
    if dataset =='kitti':
        target = batch_data['gt_sparse']
        mse_loss = MaskedMSELoss(pred, target)
        grid_loss = grid_reg*regMSE(vector_field)
        image_loss = image_reg*regSampleCoords(grid_to_sample)
        return mse_loss + grid_loss + image_loss
    elif dataset == 'nyu_v2':
        target = batch_data['gt']
        mse_loss = MSE(pred, target)
        grid_loss = grid_reg*regMSE(vector_field)
        image_loss = image_reg*regSampleCoords(grid_to_sample)
        return mse_loss + grid_loss + image_loss
    else:
        print("invalid data")
        exit()

def regMSE(vec_field):
    return torch.mean(vec_field[:,:,:,0]**2 + vec_field[:,:,:,1]**2)

def regSampleCoords(grid_to_sample):
    # return torch.mean(torch.clamp(torch.abs(grid_to_sample) - 1, min=0)**2)
    return torch.mean(grid_to_sample**2)

class logger:
    """
    Logger class for logging results and saving images from the training process. Adapted from
    Self-supervised Sparse-to-Dense https://github.com/fangchangma/self-supervised-depth-completion
    """
    def __init__(self, args):
        self.best_result = Result()
        self.best_result.set_to_worst()

        self.args = args

    def print(self, i, epoch, lr, n_set, blk_avg_meter, avg_meter):
        avg = avg_meter.average()
        blk_avg = blk_avg_meter.average()
        print(
            'Epoch: {0} [{1}/{2}]\tlr={lr} '
            'RMSE={blk_avg.rmse:.2f}({average.rmse:.2f}) '
            'MAE={blk_avg.mae:.2f}({average.mae:.2f}) '
            'iRMSE={blk_avg.irmse:.2f}({average.irmse:.2f}) '
            'iMAE={blk_avg.imae:.2f}({average.imae:.2f})\n\t'
            'silog={blk_avg.silog:.2f}({average.silog:.2f}) '
            'squared_rel={blk_avg.squared_rel:.2f}({average.squared_rel:.2f}) '
            'Delta1={blk_avg.delta1:.3f}({average.delta1:.3f}) '
            'REL={blk_avg.absrel:.3f}({average.absrel:.3f})\n\t'
            'Lg10={blk_avg.lg10:.3f}({average.lg10:.3f}) '
            .format(epoch,
                    i + 1,
                    n_set,
                    lr=lr,
                    blk_avg=blk_avg,
                    average=avg))
        blk_avg_meter.reset()

    def conditional_save_img_comparison(self, i, batch_data, pred, epoch, dataset):
        if dataset == 'nyu_v2':
            valid_mask = batch_data['gt'] > 0
            abs_diff = (pred.data[valid_mask] - batch_data['gt'].data[valid_mask]).abs()
            mse = float((torch.pow(abs_diff, 2)).mean())
            rmse = math.sqrt(mse)
        if dataset == 'kitti':
            valid_mask = batch_data['gt_sparse'] > 0
            abs_diff = (pred.data[valid_mask] - batch_data['gt_sparse'].data[valid_mask]).abs()
            abs_diff = abs_diff * 1e3
            mse = float((torch.pow(abs_diff, 2)).mean())
            rmse = math.sqrt(mse)

        if dataset == 'nyu_v2':
            # center crop the data, (borders of NYU are not image)
            batch_data['rgb'] = crop(F.interpolate(batch_data['rgb'],
                                              scale_factor=0.5), 228, 304)
            batch_data['gt'] = crop(batch_data['gt'], 228, 304)
            batch_data['d'] = crop(batch_data['d'], 228, 304)
            pred = crop(pred, 228, 304)
            skip = 3
        else:
            batch_data['gt'] = batch_data['gt_sparse']
            skip = 5
        if i == 0:
            self.img_merge = merge_into_row(batch_data['rgb']*255, pred,
                                            batch_data['gt'], batch_data['d'])
            self.RMSEVals = [rmse]
        elif i % skip == 0 and i < 160 * skip:
            row = merge_into_row(batch_data['rgb']*255, pred,
                                 batch_data['gt'], batch_data['d'])
            self.img_merge = add_row(self.img_merge, row)
            self.RMSEVals.append(rmse)
        elif i == 160 * skip:
            filename = os.path.join(self.args.save_directory,
                                    "comparison_" + str(epoch) + ".png")
            save_image(self.img_merge, filename)
            sio.savemat(os.path.join(self.args.save_directory,
                                     "validation_" + str(epoch) + ".mat"),
                        {'rmses': self.RMSEVals})

    def rank_save_best(self, result, epoch):
        error = getattr(result, 'rmse')
        best_error = getattr(self.best_result, 'rmse')
        is_best = error < best_error
        if is_best:
            self.old_best_result = self.best_result
            self.best_result = result
            self.save_best_txt(result, epoch)
        return is_best

    def save_single_txt(self, filename, result, epoch):
        with open(filename, 'w') as txtfile:
            txtfile.write(
                ("epoch={}\n" + "rmse={:.3f}\n" +
                 "mae={:.3f}\n" + "silog={:.3f}\n" + "squared_rel={:.3f}\n" +
                 "irmse={:.3f}\n" + "imae={:.3f}\n" + "mse={:.3f}\n" +
                 "absrel={:.3f}\n" + "lg10={:.3f}\n" + "delta1={:.3f}\n"
                 ).format(epoch, result.rmse, result.mae, result.silog,
                          result.squared_rel, result.irmse,
                          result.imae, result.mse, result.absrel,
                          result.lg10, result.delta1))

    def save_best_txt(self, result, epoch):
        self.save_single_txt(os.path.join(self.args.save_directory, 'best.txt'),
                             result,
                             epoch)

    def summarize(self, avg, is_best):
        print("\n*\nSummary of round")
        print(''
              'RMSE={average.rmse:.3f}\n'
              'MAE={average.mae:.3f}\n'
              'iRMSE={average.irmse:.3f}\n'
              'iMAE={average.imae:.3f}\n'
              'squared_rel={average.squared_rel}\n'
              'silog={average.silog}\n'
              'Delta1={average.delta1:.3f}\n'
              'REL={average.absrel:.3f}\n'
              'Lg10={average.lg10:.3f}'.format(average=avg))
        if is_best:
            print("New best model by rmse (was %.3f)" %
                  (getattr(self.old_best_result, 'rmse')))
        else:
            print("(best rmse is %.3f)" %
                  (getattr(self.best_result, 'rmse')))
        print("*\n")


def merge_into_row(rgb, pred, gt, sp_d):
    def preprocess_depth(x):
        y = np.squeeze(x.data.cpu().numpy())
        return depth_colorize(y)

    img_list = []
    rgb = np.squeeze(rgb[0, ...].data.cpu().numpy())
    rgb = np.transpose(rgb, (1, 2, 0))
    img_list.append(rgb)

    img_list.append(preprocess_depth(pred[0, ...]))
    img_list.append(preprocess_depth(gt[0, ...]))
    img_list.append(preprocess_depth(sp_d[0, ...]))


    img_merge = np.hstack(img_list)
    return img_merge.astype('uint8')


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')


def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)


lg_e_10 = math.log(10)


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / lg_e_10


class Result(object):
    """
    Single Result object for quantifying performance, adapted from Self-supervised
    Sparse-to-Dense repository
    """
    def __init__(self):
        self.irmse = 0
        self.imae = 0
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.absrel = 0
        self.squared_rel = 0
        self.lg10 = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.silog = 0  # Scale invariant logarithmic error [log(m)*100]

    def set_to_worst(self):
        self.irmse = np.inf
        self.imae = np.inf
        self.mse = np.inf
        self.rmse = np.inf
        self.mae = np.inf
        self.absrel = np.inf
        self.squared_rel = np.inf
        self.lg10 = np.inf
        self.silog = np.inf
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10, \
            delta1, delta2, delta3, silog):
        self.irmse = irmse
        self.imae = imae
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.absrel = absrel
        self.squared_rel = squared_rel
        self.lg10 = lg10
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.silog = silog

    def evaluate(self, output, target):
        valid_mask = target > 0.1

        # convert from meters to mm
        output_mm = 1e3 * output[valid_mask]
        target_mm = 1e3 * target[valid_mask]

        abs_diff = (output_mm - target_mm).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output_mm) - log10(target_mm)).abs().mean())
        self.absrel = float((abs_diff / target_mm).mean())
        self.squared_rel = float(((abs_diff / target_mm)**2).mean())

        maxRatio = torch.max(output_mm / target_mm, target_mm / output_mm)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25**2).float().mean())
        self.delta3 = float((maxRatio < 1.25**3).float().mean())

        # silog uses meters
        err_log = torch.log(target[valid_mask]) - torch.log(output[valid_mask])
        normalized_squared_log = (err_log**2).mean()
        log_mean = err_log.mean()
        self.silog = math.sqrt(normalized_squared_log -
                               log_mean * log_mean) * 100

        # convert from meters to km
        inv_output_km = (1e-3 * output[valid_mask])**(-1)
        inv_target_km = (1e-3 * target[valid_mask])**(-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())


class AverageMeter(object):
    """AverageMeter object for keeping track of reconstruction statistics, adapted from
    Sparse-to-Dense implementation"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_irmse = 0
        self.sum_imae = 0
        self.sum_mse = 0
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_absrel = 0
        self.sum_squared_rel = 0
        self.sum_lg10 = 0
        self.sum_delta1 = 0
        self.sum_delta2 = 0
        self.sum_delta3 = 0
        self.sum_silog = 0

    def update(self, result, n=1):
        self.count += n
        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_squared_rel += n * result.squared_rel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_silog += n * result.silog

    def average(self):
        avg = Result()
        if self.count > 0:
            avg.update(
                self.sum_irmse / self.count, self.sum_imae / self.count,
                self.sum_mse / self.count, self.sum_rmse / self.count,
                self.sum_mae / self.count, self.sum_absrel / self.count,
                self.sum_squared_rel / self.count, self.sum_lg10 / self.count,
                self.sum_delta1 / self.count, self.sum_delta2 / self.count,
                self.sum_delta3 / self.count, self.sum_silog / self.count)
        return avg


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    """differentiable SSIM implementation from DenseDepth, https://github.com/ialhashim/DenseDepth"""
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs

    return ret

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def msssim(img1, img2, val_range, window_size=11, size_average=True, normalize=False):
    """differentiable SSIM, implementation from https://github.com/serkansulun/pytorch-msssim/blob/master/pytorch_msssim.py"""
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, val_range=val_range, window_size=window_size, size_average=size_average, full=True)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

def crop(img, tw, th):
    w, h = img.shape[2:]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return img[:, :, x1:x1 + tw, y1:y1 + th]

def generate_poissondisk_mask(r, dataset='kitti', samples=None):
    if dataset == 'kitti':
        mask = np.zeros((1, 1, 352, 1216))
    else:
        mask = np.zeros((1, 1, 240, 320))
    while True:
        if dataset == 'kitti':
            i1, i2 = poisson_disc_samples(352, 1216, r=r)
        else:
            i1, i2 = poisson_disc_samples(240, 320, r=r)

        # conditions to break, make sure don't return 0 samples due to bad
        # initialization or the number of samples generated not too far or
        # close to the desired number
        if samples is not None and len(i1) >= .9 * samples and len(i1) <= 1.1 * samples:
            break
        elif len(i1) >= 50 and dataset =='kitti':
            break
        elif len(i1) >= 30 and dataset =='nyu_v2':
            break
    mask[0, 0, i1, i2] = 1
    mask = torch.from_numpy(mask)
    return mask

def euclidean_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return sqrt(dx * dx + dy * dy)

def poisson_disc_samples(width, height, r, k=5, distance=euclidean_distance, random=random):
    '''
    method for generating poisson disc samples, adapted from https://github.com/emulbreh/bridson
    '''
    tau = 2 * pi
    cellsize = r / sqrt(2)

    grid_width = int(ceil(width / cellsize))
    grid_height = int(ceil(height / cellsize))
    grid = [None] * (grid_width * grid_height)

    def grid_coords(p):
        return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

    def fits(p, gx, gy):
        yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in yrange:
                g = grid[x + y * grid_width]
                if g is None:
                    continue
                if distance(p, g) <= r:
                    return False
        return True

    p = width * random(), height * random()
    queue = [p]
    grid_x, grid_y = grid_coords(p)
    grid[grid_x + grid_y * grid_width] = p

    while queue:
        qi = int(random() * len(queue))
        qx, qy = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            alpha = tau * random()
            d = r * sqrt(3 * random() + 1)
            px = qx + d * cos(alpha)
            py = qy + d * sin(alpha)
            if not (0 <= px < width and 0 <= py < height):
                continue
            p = (px, py)
            grid_x, grid_y = grid_coords(p)
            if not fits(p, grid_x, grid_y):
                continue
            queue.append(p)
            grid[grid_x + grid_y * grid_width] = p
    # return [(min(round(p[0]), width-1), min(round(p[1]), height-1)) for p in grid if p is not None]
    # return [(round(p[0]), round(p[1])) for p in grid if p is not None]
    return [min(round(p[0]), width - 1) for p in grid if p is not None], [min(round(p[1]), height - 1) for p in grid if p is not None]

def save_methods_figure(batch_data, spd, vecfield, pred):
    """
    helper method for saving a figure of all steps of the pipeline
    """
    rgb = 255 * crop(F.interpolate(batch_data['rgb'], scale_factor=0.5),
                     228, 304)
    rgb = np.squeeze(rgb[0, ...].data.cpu().numpy())
    rgb = np.transpose(rgb, (1, 2, 0))

    mdi = crop(batch_data['mdi'], 228, 304)
    mdi = np.squeeze(mdi[0, ...].data.cpu().numpy())
    mdi = depth_colorize(mdi)

    gt = crop(batch_data['gt'], 228, 304)
    gt = np.squeeze(gt[0, ...].data.cpu().numpy())
    gt = depth_colorize(gt)

    inpainted = crop(batch_data['bproxi'], 228, 304)
    inpainted = np.squeeze(inpainted[0, ...].data.cpu().numpy())
    inpainted = depth_colorize(inpainted)

    spd = crop(spd, 228, 304)
    spd = np.squeeze(spd[0, ...].data.cpu().numpy())
    spd = depth_colorize(spd)

    pred = crop(pred, 228, 304)
    pred = np.squeeze(pred[0, ...].data.cpu().numpy())
    pred = depth_colorize(pred)

    vecfield = vecfield.data.cpu().numpy()

    sio.savemat("methods_figure.mat", {'rgb': rgb,
                                       'mdi': mdi,
                                       'gt': gt,
                                       'spd': spd,
                                       'vecfield': vecfield,
                                       'inpainted': inpainted,
                                       'pred': pred
                                       })

def generate_random_mask_kitti(samples):
    """
    generate a random sampling mask the size of kitti images
    """
    prob = float(samples) / (256*1216)
    mask_keep = np.random.uniform(0, 1, (1, 1, 352, 1216)) < prob
    return torch.from_numpy(mask_keep.astype(np.float)).float()


