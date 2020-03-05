import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

import os
import os.path


def GetKittiData(batch_size, samples, sampling_method, data_directory):
    # data_directory = '/media/data2/awb/kitti/inpainted'

    training = depthDataset(data_directory,
                            "train",
                            samples,
                            sampling_method)
    validation = depthDataset(data_directory,
                              "val_select_inpainted",
                              samples,
                              sampling_method)

    return DataLoader(training, batch_size, shuffle=True, num_workers=4), \
           DataLoader(validation, 1, shuffle=False, num_workers=4)


class depthDataset(Dataset):
    def __init__(self, directory_inpainted, split,
                 samples, sampling_method):
        self.directory_inpainted = directory_inpainted
        self.split = split
        self.samples = samples
        self.sampling_method = sampling_method

        self.transform = get_transform(split)

        paths = []
        if split == "train":
            for subdirectory in os.listdir(os.path.join(self.directory_inpainted,
                                                        split)):
                for image in os.listdir(os.path.join(self.directory_inpainted,
                                                     split,
                                                     subdirectory)):
                    if image[0] == 'd':
                        paths.append((subdirectory, image[-14:]))

            # train on a subset of the dataset
            # paths = paths[(1 * (len(paths) // 3)):(2 * (len(paths) // 3))]
            paths = paths[(2*(len(paths) // 3)):]

        elif split == "val_select_inpainted":
            for image in os.listdir(os.path.join(self.directory_inpainted,
                                                 split)):
                if image[0] == 'd':
                    paths.append(image[2:])

                paths = sorted(paths)
        else:
            print("unsupported split")
            exit()

        self.paths = paths

    def __getraw__(self, index):
        if self.split == "train":
            subdirectory, image_ID = self.paths[index]
            rgb = Image.open(os.path.join(self.directory_inpainted,
                                          self.split,
                                          subdirectory,
                                          "rgb_" + image_ID))
            depth = Image.open(os.path.join(self.directory_inpainted,
                                            self.split,
                                            subdirectory,
                                            "d_" + image_ID))
        elif self.split == "val_select_inpainted":
            image_ID = self.paths[index]
            rgb = Image.open(os.path.join(self.directory_inpainted,
                                          self.split,
                                          "rgb_" + image_ID))
            depth = Image.open(os.path.join(self.directory_inpainted,
                                            self.split,
                                            "d_" + image_ID))
        return rgb, depth

    def get_sparse_gt(self, index):
        if self.split == "train":
            subdirectory, image_ID = self.paths[index]

            path = os.path.join('/media/data2/awb/kitti/annotated/train',
                                subdirectory,
                                'proj_depth/groundtruth/image_02',
                                image_ID)
        elif self.split == "val_select_inpainted":
            image_ID = self.paths[index]

            path = os.path.join('/media/data2/awb/kitti/depth_selection/depth_selection',
                                'val_selection_cropped/groundtruth_depth',
                                image_ID[:27] + 'groundtruth_depth_' + image_ID[27:])

        return depth_read(path)

    def __getitem__(self, idx):
        rgb, gt_depth = self.__getraw__(idx)
        sparse_gt = self.get_sparse_gt(idx)

        sample = {'rgb': rgb, 'gt': gt_depth, 'gt_sparse': sparse_gt}
        sample = self.transform(sample)

        sp_depth = self.sparsify(sample['gt'])
        sample['d'] = sp_depth

        return sample

    def __len__(self):
        return len(self.paths)

    def sparsify(self, gt):
        if self.sampling_method == "u_r":
            prob = float(self.samples) / (352*1216)
            mask = np.random.rand(*gt.shape) < prob

            sparse = gt * torch.from_numpy(mask.astype(np.float)).float()

            return sparse
        else:
            print("unimplemented sampling method")
            exit()

class ToTensorKITTI(object):
    def __call__(self, sample):
        image, depth, gt_sparse = sample['rgb'], sample['gt'], sample['gt_sparse']

        # image = image.resize((1216, 352))
        # image = bottom_crop(image, (1216, 352))
        image = self.to_tensor(image)
        depth = self.to_tensor(depth).float()
        gt_sparse = torch.from_numpy(np.transpose(gt_sparse, (2, 0, 1))).float()

        image = bottom_crop(image, (352, 1216))
        depth = bottom_crop(depth, (352, 1216))
        gt_sparse = bottom_crop(gt_sparse, (352, 1216))

        # put in expected range

        return {'rgb': image, 'gt': depth, 'gt_sparse': gt_sparse}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img.float().div(256)


def get_transform(split):
    if split == "train":
        return transforms.Compose([
            RandomHorizontalFlip(),
            RandomChannelSwap(0.5),
            ToTensorKITTI()
        ])
    elif split == "val_select_inpainted":
        return transforms.Compose([
            ToTensorKITTI()
        ])
    else:
        print("unknown split")
        exit()


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth, gt_sparse = sample['rgb'], sample['gt'], sample['gt_sparse']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            gt_sparse = np.fliplr(gt_sparse).copy()

        return {'rgb': image, 'gt': depth, 'gt_sparse': gt_sparse}


class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth, gt_sparse = sample['rgb'], sample['gt'], sample['gt_sparse']
        if not _is_pil_image(image): raise TypeError(
            'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError(
            'img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(
                self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'rgb': image, 'gt': depth, 'gt_sparse': gt_sparse}


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def depth_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth

def bottom_crop(img, output_size):
    h = img.shape[1]
    w = img.shape[2]
    th, tw = output_size
    i = h - th
    j = int(round((w - tw) / 2.))

    if img.dim() == 3:
        return img[:, i:i + th, j:j + tw]
    elif img.ndim == 2:
        return img[i:i + th, j:j + tw]
