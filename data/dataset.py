import os
import glob
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from imageio import imread
from skimage.color import rgb2gray, gray2rgb
from utils import imagetools


class Dataset(torch.utils.data.Dataset):
    def __init__(self, flist, image_size):
        super(Dataset, self).__init__()
        self.data = self.load_flist(flist)

        self.input_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        # load image
        img = imread(self.data[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        img = self.resize(img, self.input_size, self.input_size)
 
        img = img/(2 ** 8 - 1)

        # masked and ref channel
        masked_image = imagetools.rgb_to_random_gray_hsv(img)
        ref_image = imagetools.rgb_to_random_gray_hsv(img)
 
        # load mask
        mask = imagetools.create_mask(img)

        return self.to_tensor(masked_image), self.to_tensor(ref_image), self.to_tensor(mask)

    def to_tensor(self, img):
        img_t = torch.tensor(img[None, :, :]).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if img.shape[0] == height and img.shape[1] == width:
            return img

        img = Image.fromarray(img).resize(size=(height, width))

        return np.array(img)

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
