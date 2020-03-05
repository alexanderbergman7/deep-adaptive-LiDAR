import os.path
from torch.utils.data import Dataset, DataLoader

import os.path
import dataloaders.transforms_nyu as transforms

import os
import os.path
import numpy as np
import h5py

# THIS FILE IS CURRENTLY UNUSED. IT IS BASED ON THE DATALOADING MECHANISM IN
# https://github.com/fangchangma/self-supervised-depth-completion


def GetNYUV2Data(batch_size, samples, sampling_method, data_directory):
    # data_directory = '/media/data2/awb/NYU_DEPTH_V2/' \
    #                  'nyudepthv2_std_orig'

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

        classes, class_to_idx = find_classes(os.path.join(self.directory,
                                                          self.split))
        imgs = make_dataset(os.path.join(self.directory,
                                         self.split), class_to_idx)

        assert len(imgs)>0, "Found 0 images in subfolders of: "\
                            + os.path.join(self.directory, self.split)\
                            + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))

        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx

        if split == 'train':
            self.transform = self.train_transform
        elif split == 'val':
            self.transform = self.val_transform
        else:
            print("invalid split")
            exit()

        self.output_size = (228, 304)

    def __getraw__(self, index):
        path, target = self.imgs[index]
        rgb, depth = h5_loader(path)
        return rgb, depth

    def __getitem__(self, idx):
        rgb, depth = self.__getraw__(idx)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise(RuntimeError("transform not defined"))

        rgb = to_tensor(rgb_np)
        d = to_tensor(self.sparsify(depth_np))
        while rgb.dim() < 3:
            rgb = rgb.unsqueeze(0)
        while d.dim() < 3:
            d = d.unsqueeze(0)
        gt = to_tensor(depth_np)
        gt = gt.unsqueeze(0)

        return {'rgb': rgb, 'd': d, 'gt': gt}

    def __len__(self):
        return len(self.imgs)

    def sparsify(self, depth):
        if self.sampling_method == "u_r":
            mask_keep = depth > 0
            n_keep = np.count_nonzero(mask_keep)
            prob = float(self.samples) / n_keep
            mask_keep = np.bitwise_and(mask_keep,
                                       np.random.uniform(0, 1, depth.shape) < prob)
            sparse = depth * mask_keep

            return sparse
        else:
            print("unimplemented sampling method")
            exit()

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        transform = transforms.Compose([
            # transforms.Resize(250.0 / iheight),
            # transforms.Rotate(angle),
            # transforms.Resize(s),
            # transforms.CenterCrop(self.output_size),
            transforms.Resize(240.0 / iheight),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        rgb_np = color_jitter(rgb_np)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255.0
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            # transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255.0
        depth_np = transform(depth_np)

        return rgb_np, depth_np

color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
IMG_EXTENSIONS = ['.h5',]
to_tensor = transforms.ToTensor()
iheight, iwidth = 480, 640


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth
