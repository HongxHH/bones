import torch.utils.data as data
import os.path
import cv2
import numpy as np
from data import common

def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]

def npy_loader(path):
    return np.load(path)


IMG_EXTENSIONS = [
    '.png', '.npy',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)): # 遍历dir目录下的所有文件和子目录
        for fname in fnames: # 遍历当前目录下的所有文件
            if is_image_file(fname):
                path = os.path.join(root, fname) # 将当前文件的绝对路径添加到images列表中
                images.append(path) # 将当前文件的绝对路径添加到images列表中
    return images


# training dataset for overfitting content-aware SR models
class OverfitDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.scale = self.opt.scale
        self.root = self.opt.root
        self.ext = self.opt.ext
        self.train = True if self.opt.phase == 'train' else False
        # self._set_filesystem()

        self.dir_hr = self.opt.dir_hr
        self.dir_lr = self.opt.dir_lr
        
        self.images_hr, self.images_lr = self._scan()
        self.num_samples = min(len(self.images_hr), len(self.images_lr))
        if self.num_samples == 0:
            raise RuntimeError(f"No training pairs found under {self.dir_lr} and {self.dir_hr}")
        # 如果未指定 n_train，则默认覆盖全部样本
        if getattr(self.opt, "n_train", None) in (None, 0):
            self.opt.n_train = self.num_samples

    # def _set_filesystem(self):
    #     scale2res = {2:"480p", 3:"360p", 4:"240p"}
    #     self.dir_hr = os.path.join(self.root, '1080p-0.1fps')
    #     self.dir_lr = os.path.join(self.root, "{}-0.1fps-no-upscale".format(scale2res[self.scale]))

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.opt.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr, rgb_range=self.opt.rgb_range)
        return lr_tensor, hr_tensor

    def __len__(self):
        return self.num_samples

    def _get_index(self, idx):
        return idx % self.num_samples

    def _get_patch(self, img_in, img_tar):
        patch_size = self.opt.patch_size
        scale = self.scale
        if self.train:
            img_in, img_tar = common.get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = common.augment(img_in, img_tar)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
        return lr, hr