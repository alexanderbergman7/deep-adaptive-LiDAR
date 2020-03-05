import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numbers
import math
import random
import utils

import models.pytorch_prototyping_orig as pytorch_prototyping_orig

'''Module containing the network for predicting the sparse sampling locations and differentiably
sampling at these locations'''

class SparseDepthPrediction(nn.Module):
    def __init__(self, samplesH, samplesW, shape, device=None, ret_samples=False,
                 dataset='kitti'):
        super(SparseDepthPrediction, self).__init__()
        self.device = device
        self.samplesH = samplesH
        self.samplesW = samplesW
        self.shape = shape

        self.UNET = pytorch_prototyping_orig.Unet(in_channels=1,
                                                  out_channels=2,
                                                  nf0=2,
                                                  num_down=4,
                                                  max_channels=512,
                                                  use_dropout=False,
                                                  outermost_linear=True)

        self.xx_channel, self.yy_channel = self.get_coord_feature(shape)
        self.xx_channel_eval, self.yy_channel_eval = \
            self.get_coord_feature((1, shape[1], shape[2], shape[3]))

        self.ret_samples = ret_samples
        self.dataset = dataset


    def forward(self, batch_data):
        # Uncomment if desired to have the pixel coordinate as features. Performance is similar
        # Note: need to edit the self.UNET in_channels above to 3

        # if self.training:
        #     monocular_features = torch.cat((batch_data['mdi'],
        #                                     self.xx_channel, self.yy_channel), dim=1)
        # else:
        #     monocular_features = torch.cat((
        #         batch_data['mdi'], self.xx_channel_eval, self.yy_channel_eval), dim=1)

        # Otherwise, use the monocular depth image as input to prediction
        monocular_features = batch_data['mdi']

        dense_vector_field = self.UNET(monocular_features)

        if self.training:
            samplesH = self.samplesH + random.randint(-3,3)
            samplesW = self.samplesW + random.randint(-9,9)
        else:
            samplesH = self.samplesH + random.randint(-1,1)
            samplesW = self.samplesW + random.randint(-3,3)

        # smoothing the vector field and then downsampling (based on nearest neighbor) is equivalent
        # to the vector at acoordinate being the sum of the nearby pixels
        smoothed = self.GaussianSmooth(dense_vector_field,
                                       channels=2,
                                       kernel_size=[int(2*self.shape[2]/(samplesH+1)),
                                                    int(2*self.shape[3]/(samplesW+1))],
                                       sigma=[2*self.shape[2]/(samplesH+1)/3,
                                              2*self.shape[3]/(samplesW+1)/3])

        sparse_vector_field = F.interpolate(smoothed, size=(samplesH, samplesW),
                                            mode='nearest')

        if self.ret_samples:
            sparse_d, grid_to_sample, samps = self.WarpSample(batch_data,
                                                       sparse_vector_field.permute(0,2,3,1),
                                                       samplesW, samplesH)
            return sparse_d, dense_vector_field.permute(0, 2, 3, 1), grid_to_sample, samps

        sparse_d, grid_to_sample = self.WarpSample(batch_data,
                                                   sparse_vector_field.permute(0,2,3,1),
                                                   samplesW, samplesH)
        return sparse_d, dense_vector_field.permute(0, 2, 3, 1), grid_to_sample

    def get_coord_feature(self, shape):
        '''
        Return the coordinate features to be concatenated to an image
        '''
        xx_ones = torch.ones([1, shape[3]], dtype=torch.int32, device=self.device).float()
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(shape[2], dtype=torch.int32, device=self.device).unsqueeze(0).float()
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones, xx_range)
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, shape[2]], dtype=torch.int32, device=self.device).float()
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(shape[3], dtype=torch.int32, device=self.device).unsqueeze(0).float()
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range, yy_ones)
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        xx_channel = xx_channel / (shape[2] - 1)
        yy_channel = yy_channel / (shape[3] - 1)

        xx_channel = xx_channel.repeat(shape[0], 1, 1, 1)
        yy_channel = yy_channel.repeat(shape[0], 1, 1, 1)

        return xx_channel, yy_channel

    def GaussianSmooth(self, input, channels, kernel_size, sigma):
        '''
        Gaussian smooth some input image
        '''
        dim = 2
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        return F.conv2d(input, weight=kernel.to(self.device), groups=channels)

    def WarpSample(self, batch_data, sparse_vector_field, samplesW, samplesH):
        grid_w, grid_h = np.meshgrid(np.linspace(-((samplesW - 1) / samplesW),
                                                 ((samplesW - 1) / samplesW),
                                                 samplesW),
                                     np.linspace(-((samplesH - 1) / samplesH),
                                                 ((samplesH - 1) / samplesH),
                                                 samplesH))
        grid_w = torch.Tensor(grid_w)
        grid_h = torch.Tensor(grid_h)
        norm_grid = torch.stack((grid_w,grid_h),2).to(self.device)

        # broadcast self.norm_grid over
        grid_to_sample = sparse_vector_field + norm_grid

        gt_sampled = F.grid_sample(batch_data['gt'], grid_to_sample)
        sparse = self.expand(gt_sampled, grid_to_sample, samplesW, samplesH)

        # uncomment for random sampling, or poisson disk sampling instaed of the adaptive sampling
        # this should only be used for benchmarking, note we hack the samplesH variable to represent the
        # number of samples or the radius of the poisson disc sampling
        # sparse = utils.generate_poissondisk_mask(r=17.6).float().to(self.device) * batch_data['gt']
        # sparse = utils.generate_random_mask_kitti(samplesH).float().to(self.device) * batch_data['gt']

        if not self.training and self.dataset == 'kitti':
            sparse[:, :, 0:96, :] = 0

        if self.ret_samples:
            samps = np.count_nonzero(sparse.detach().cpu().numpy())
            return sparse, grid_to_sample, samps

        return sparse, grid_to_sample

    def expand(self, gt_sampled, grid_to_sample, samplesW, samplesH):
        sparse_img = torch.zeros(gt_sampled.shape[0],
                                 gt_sampled.shape[1],
                                 self.shape[2],
                                 self.shape[3],
                                 device=self.device)

        for b in range(gt_sampled.shape[0]):
            grid_wp = grid_to_sample[b,:,:,0]  # 1 x 6 x 8 x 1
            grid_hp = grid_to_sample[b,:,:,1]  # 1 x 6 x 8 x 1
            rc = 0
            cc = 0
            for row,col in zip(grid_hp.view(-1).detach().cpu().numpy().tolist(),
                               grid_wp.view(-1).detach().cpu().numpy().tolist()):
                sparse_img[b,
                           :,
                           max(min(int(((row+1)/2)*self.shape[2]), self.shape[2]-1),0),
                           max(min(int(((col+1)/2)*self.shape[3]), self.shape[3]-1),0)] = \
                    gt_sampled[b,:,int(rc), int(cc)]
                cc += 1
                cc = cc % samplesW
                rc += (1/samplesW)

        return sparse_img
