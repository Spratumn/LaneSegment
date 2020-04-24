import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2 as cv
from torch.utils.data import Dataset
from imgaug import augmenters as iaa

from utils.label_process import label_encoder
from config import Config

cfg = Config()


# attention: .jpg read by opencv,  .png read by PIL

# crop the image to discard useless parts
def crop_resize_data(image, label=None):
    """
    Attention:
    h,w, c = image.shape
    cv2.resize(image,(w,h))
    """
    roi_image = image[cfg.CROP_SIZE:, :]  # crop size
    _, h, w = cfg.IMAGE_SHAPE
    image_size = (int((h+1-cfg.CROP_SIZE)/cfg.RESIZE_SCALE), int(w/cfg.RESIZE_SCALE))
    if label is not None:
        roi_label = label[cfg.CROP_SIZE:, :]
        train_image = cv.resize(roi_image, image_size, interpolation=cv.INTER_LINEAR)
        train_label = cv.resize(roi_label, image_size, interpolation=cv.INTER_NEAREST)
        return train_image, train_label
    else:
        train_image = cv.resize(roi_image, image_size, interpolation=cv.INTER_LINEAR)
        return train_image


class LaneSegTrainDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        super(LaneSegTrainDataset, self).__init__()
        # read csv as data index
        self.data_paths = pd.read_csv(os.path.join(cfg.CSV_DIR, csv_file),
                                      header=None,
                                      names=["image", "label"])
        self.image_paths = self.data_paths["image"].values[1:]
        self.label_paths = self.data_paths["label"].values[1:]
        self.transform = transform

    def __len__(self):
        return self.label_paths.shape[0]

    def __getitem__(self, idx):
        # read image and label by index
        ori_image = cv.imread(self.image_paths[idx])
        ori_label = np.array(Image.open(self.label_paths[idx]))
        # crop and resize image and label
        train_img, ori_label = crop_resize_data(ori_image, ori_label)
        # Encode
        train_label = label_encoder(ori_label)

        sample = {'image': train_img,
                  'label': train_label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # opencv image: H x W x C
        # torch image: C X W X H
        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image.astype(np.float32))
        label_tensor = torch.from_numpy(label.astype(np.long))
        return {'image': image_tensor,
                'label': label_tensor}


# imgaug Augmentation
class ImageAug(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.uniform(0, 1) > 0.5:
            seq = iaa.Sequential([iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)),
                iaa.GaussianBlur(sigma=(0, 1.0))])])
            image = seq.augment_image(image)
        return {'image': image,
                'label': label}


class LaneSegTestDataset(Dataset):
    def __init__(self, csv_file):
        super(LaneSegTestDataset, self).__init__()
        # read csv as data index
        self.data_paths = pd.read_csv(os.path.join(cfg.CSV_DIR, csv_file),
                                      header=None,
                                      names=["image", "label"])
        self.image_paths = self.data_paths["image"].values[1:]
        self.label_paths = self.data_paths["label"].values[1:]

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, idx):
        # read image and label by index
        ori_img_path = self.image_paths[idx]
        ori_image = cv.imread(self.image_paths[idx])
        if cfg.INFER_MODE == 'eval':
            ori_label = np.array(Image.open(self.label_paths[idx]))
            train_img, ori_label = crop_resize_data(ori_image, ori_label)
            train_label = label_encoder(ori_label)
            train_img = train_img.transpose((2, 0, 1))
            image_tensor = torch.from_numpy(train_img.astype(np.float32))
            label_tensor = torch.from_numpy(train_label.astype(np.long))
            return {'image': image_tensor,
                    'label': label_tensor,
                    'img_path': ori_img_path}
        else:
            train_img = crop_resize_data(ori_image)
            train_img = train_img.transpose((2, 0, 1))
            image_tensor = torch.from_numpy(train_img.astype(np.float32))
            return {'image': image_tensor,
                    'img_path': ori_img_path}
