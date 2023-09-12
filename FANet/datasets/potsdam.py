import os
import random
import numpy as np
from PIL import Image
from torch.utils import data
import skimage.io as ski
num_classes = 2
ignore_label = 255

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'Data'))

def calWhitepercent(data):
    out = np.sum(np.any(data==[255, 255, 255], axis=-1)) / np.float(data.size)
    return out

def make_dataset(mode):
    assert (mode in ['train', 'val'])

    img_path = os.path.join(root, mode, 'image')
    mask_path = os.path.join(root, mode, 'label')
    #print(os.listdir(img_path))
    #print(os.listdir(mask_path))

    assert os.listdir(img_path) == os.listdir(mask_path)
    items = []
    
    if(mode == 'train'):
        c_items = random.sample(os.listdir(img_path), 10000)#random choice 10000 images every epoch
    else:
        c_items = (os.listdir(img_path))

    for it in c_items:
        item = (os.path.join(img_path, it), os.path.join(mask_path, it))
        #img = ski.imread(item[0])
        #img = np.array(Image.open(img_path))
        #if calWhitepercent(img) > 0.2:
        #    continue
        items.append(item)
    return items


class Potsdam(data.Dataset):
    def __init__(self,mode, simul_transform=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.mode = mode
        self.simul_transform = simul_transform
        self.transform = transform
        self.target_transform = target_transform
        self.id_to_trainid = {0: 0, 1: 1}

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.simul_transform is not None:
            img, mask = self.simul_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)
