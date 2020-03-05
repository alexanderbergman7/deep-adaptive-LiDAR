import os.path
from torch.utils.data import Dataset, DataLoader

import os
import os.path
import glob
import numpy as np
from PIL import Image
import dataloaders.transforms_kitti as transforms


def GetKittiData(batch_size, samples, sampling_method, data_directory):
    # data_directory = '/media/data2/awb/kitti/' \
    #                  'sparse-to-dense/self-supervised-depth-completion/data'

    training = depthDataset(data_directory,
                            "train",
                            samples,
                            sampling_method)
    validation = depthDataset(data_directory,
                              "val",
                              samples,
                              sampling_method)

    return DataLoader(training, batch_size, shuffle=True, num_workers=4), \
           DataLoader(validation, 1, shuffle=False, num_workers=4)


class depthDataset(Dataset):
    def __init__(self, directory_imgs, split,
                 samples, sampling_method):
        self.directory = directory_imgs
        self.split = split
        self.samples = samples
        self.sampling_method = sampling_method

        self.paths, self.transform = get_paths_and_transform(split,
                                                             self.directory)

        if split == "train":
            # uncomment to train on a full data instead of subset
            length = int(len(self.paths["rgb"])/3)
            self.paths["rgb"] = self.paths["rgb"][length:-length]
            self.paths["d"] = self.paths["d"][length:-length]
            self.paths["gt"] = self.paths["gt"][length:-length]

            # self.paths["rgb"] = self.paths["rgb"]
            # self.paths["d"] = self.paths["d"]
            # self.paths["gt"] = self.paths["gt"]

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index])
        sparse = depth_read(self.paths['d'][index])
        target = depth_read(self.paths['gt'][index])
        return rgb, sparse, target

    def __getitem__(self, idx):
        rgb, sparse, target = self.__getraw__(idx)
        rgb, sparse, target = self.transform(rgb, sparse, target)

        if self.sampling_method == "none":
            sample = {'rgb': rgb, 'd': sparse, 'gt': target}
        else:
            sampling_mask, sparse = self.sparsify(sparse)
            sample = {'rgb': rgb, 'd': sparse, 'gt': target, 'mask': sampling_mask}

        sample = {
            key: to_float_tensor(val)
            for key, val in sample.items()
        }
        sample['rgb'] = sample['rgb'] / 255.0
        return sample

    def __len__(self):
        return len(self.paths['gt'])

    def sparsify(self, sparse):
        if self.sampling_method == "u_r":
            tot_samples = np.count_nonzero(sparse)

            prob = float(self.samples) / float(tot_samples)
            mask = np.random.rand(*sparse.shape) < prob

            sparse = sparse * mask

            return mask.astype(np.float), sparse
        else:
            print("unimplemented sampling method")
            exit()


def get_paths_and_transform(split, data_folder):
    if split == "train":
        transform = train_transform
        glob_d = os.path.join(
            data_folder,
            'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        )
        glob_gt = os.path.join(
            data_folder,
            'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
        )

        def get_rgb_paths(p):
            ps = p.split('/')
            pnew = '/'.join([data_folder] + ['data_rgb'] + ps[-6:-4] +
                            ps[-2:-1] + ['data'] + ps[-1:])
            return pnew
    elif split == "val":
        transform = no_transform
        glob_d = os.path.join(
            data_folder,
            "depth_selection/val_selection_cropped/velodyne_raw/*.png")
        glob_gt = os.path.join(
            data_folder,
            "depth_selection/val_selection_cropped/groundtruth_depth/*.png"
        )
        def get_rgb_paths(p):
            return p.replace("groundtruth_depth", "image")
    else:
        raise ValueError("Unrecognized split " + str(split))

    paths_d = sorted(glob.glob(glob_d))
    paths_gt = sorted(glob.glob(glob_gt))
    paths_rgb = [get_rgb_paths(p) for p in paths_gt]

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise (RuntimeError("Produced different sizes for datasets"))

    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    return paths, transform


def no_transform(rgb, sparse, target):
    return rgb, sparse, target


def val_transform(rgb, sparse, target):
    transform = transforms.Compose([
        transforms.BottomCrop((352, 1216)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    return rgb, sparse, target


def train_transform(rgb, sparse, target):
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    transform_geometric = transforms.Compose([
        transforms.BottomCrop((352, 1216)),
        transforms.HorizontalFlip(do_flip)
    ])
    if sparse is not None:
        sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - 0.1),
                                       1 + 0.1)
        contrast = np.random.uniform(max(0, 1 - 0.1), 1 + 0.1)
        saturation = np.random.uniform(max(0, 1 - 0.1),
                                       1 + 0.1)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)

    return rgb, sparse, target


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png


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


to_tensor = transforms.ToTensor()


to_float_tensor = lambda x: to_tensor(x).float()