###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_thermal_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    dir_tv = os.path.join(dir, 'TV')
    dir_ir = os.path.join(dir, 'IR_32')
    dir_label = os.path.join(dir, 'LABEL_TEMP')

    for root, _, fnames in sorted(os.walk(dir_tv)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path_tv = os.path.join(root, fname)
                fn, file_extension = os.path.splitext(fname)
                path_ir = os.path.join(dir_ir, fn + '.npy')
                path_label = os.path.join(dir_label, fn + '.npy')
                images.append({'A' : path_tv, 'B' : path_ir, 'LABEL' : path_label})

    return images

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
