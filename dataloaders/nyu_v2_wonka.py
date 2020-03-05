import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from io import BytesIO
import random

# Dataloader code adapted from https://github.com/ialhashim/DenseDepth

def GetNYUV2Data(batch_size, samples, sampling_method, data_directory):
    # data_directory = '/media/data2/awb/NYU_DEPTH_V2/' \
    #                  'nyudepthv2_densedepth/nyu_data.zip'
    # data_directory = '/media/data2/awb/nyu_v2/densedepth/nyu_data_wonka.zip'

    data, nyu2_train, nyu2_val = loadZipToMem(data_directory)

    # uncomment one of the following lines to only train on a subset of the data
    training = depthDatasetMemory(data,
                                  # nyu2_train[16967:-16868],
                                  nyu2_train[-16888:],
                                  # nyu2_train[16967:],
                                  # nyu2_train,
                                  transform=getDefaultTrainTransform(),
                                  samples=samples,
                                  sampling_method=sampling_method)
    validation = depthDatasetMemory(data,
                                    nyu2_val,
                                    transform=getNoTransform(is_test=True),
                                    samples=samples,
                                    sampling_method=sampling_method)

    return DataLoader(training, batch_size, shuffle=True, num_workers=4), \
           DataLoader(validation, 1, shuffle=False, num_workers=4)


def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list(
        (row.split(',') for row in (data['data/nyu2_train.csv']).decode(
            "utf-8").split('\n') if len(row) > 0))
    nyu2_val = list(
        (row.split(',') for row in (data['data/nyu2_test.csv']).decode(
            "utf-8").split('\n') if len(row) > 0))

    print('Loaded ({0})({1}).'.format(len(nyu2_train), len(nyu2_val)))
    return data, nyu2_train, nyu2_val


class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_split, samples, sampling_method,
                 transform=None):
        self.data, self.nyu_dataset = data, nyu2_split
        self.transform = transform
        self.samples = samples
        self.sampling_method = sampling_method

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))
        sample = {'rgb': image, 'd': depth}
        if self.transform: sample = self.transform(sample)
        sample = self.sparsify(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)

    def sparsify(self, sample):
        if self.sampling_method == "u_r":
            sample['gt'] = sample['d']
            prob = float(self.samples) / np.prod(sample['d'].shape[-2:])
            mask_keep = np.random.uniform(0, 1, sample['d'].shape) < prob
            sample['d'] = sample['d'] * torch.from_numpy(mask_keep.astype(np.float)).float()

            return sample
        else:
            print("unimplemented sampling method")
            exit()


def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])


def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['rgb'], sample['d']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'rgb': image, 'd': depth}


class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['rgb'], sample['d']
        if not _is_pil_image(image): raise TypeError(
            'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError(
            'img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(
                self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'rgb': image, 'd': depth}


class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['rgb'], sample['d']

        image = self.to_tensor(image)

        depth = depth.resize((320, 240))
        depth = self.to_tensor(depth).float()

        if self.is_test:
            depth = depth / 1000
        else:
            depth = depth * 10

        # put in expected range
        # depth = torch.clamp(depth, 10, 1000)

        return {'rgb': image, 'd': depth}

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
            return img


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})
