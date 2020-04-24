from tqdm import tqdm
import cv2 as cv
import numpy as np
import pandas as pd
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.deeplabv3p import Deeplabv3plus
from model.unet import UNetv1
from utils.image_process import LaneSegTestDataset
from utils.label_process import label_color_decoder, label_id_decoder
from utils.metric import update_confusion_matrix, compute_iou, \
                         compute_precision, compute_recall, compute_mean
from config import Config


cfg = Config()


def get_img_list(csv_name):
    """
    implement in test mode
    cfg.SRC_IMG = '/***/***'
    """
    assert os.path.exists(cfg.SRC_IMG)
    img_list = []
    for img_name in os.listdir(cfg.SRC_IMG):
        if img_name.endswith('.jpg'):
            img_list.append(os.path.join(cfg.SRC_IMG, img_name))
    img_list = pd.DataFrame({'image': img_list})
    img_list.to_csv(os.path.join(cfg.CSV_DIR, csv_name), index=False)


def label_output(prediction, save_path, img_name):
    """
    prediction: (w1,h1)
    img_path : otput path
    """
    c, h, w = cfg.IMAGE_SHAPE
    if cfg.LABEL_STYLE == 'color' or cfg.LABEL_STYLE == 'both':
        color_label_path = os.path.join(save_path, 'color_label')
        if not os.path.exists(color_label_path):
            os.makedirs(color_label_path)
        color_label_name = img_name.replace('.jpg', '_bin.png')
        
        color_label = np.zeros((c, w, h))  # (3,w,h)
        out_color_label = label_color_decoder(prediction)  # (3,w1,h1)
        out_color_label = cv.resize(out_color_label,
                                    (3, w, h-cfg.CROP_SIZE),
                                    interpolation=cv.INTER_NEAREST)  # (3,w,h2)
        color_label[:, :, cfg.CROP_SIZE:] = out_color_label
        color_label = color_label.transpose((2, 1, 0))  # (3,w,h) -> (h,w,3)
        cv.imwrite(os.path.join(color_label_path, color_label_name), color_label)
        
    elif cfg.LABEL_STYLE == 'gray' or cfg.LABEL_STYLE == 'both':
        gray_label_path = os.path.join(save_path, 'gray_label')
        if not os.path.exists(gray_label_path):
            os.makedirs(gray_label_path)
        gray_label_name = img_name.replace('.jpg', '_bin.png')

        gray_label = np.zeros((w, h))  # (w,h)
        out_gray_label = label_id_decoder(prediction)  # (w1,h1)
        out_gray_label = cv.resize(out_gray_label,
                                   (w, h-cfg.CROP_SIZE),
                                   interpolation=cv.INTER_NEAREST)
        gray_label[:, cfg.CROP_SIZE:] = out_gray_label
        gray_label = gray_label.transpose((1, 0))  # (w,h) -> (h,w)
        cv.imwrite(os.path.join(gray_label_path, gray_label_name), gray_label)
    else:
        raise ValueError('Please check the inference config parameters:"LABEL_STYLE"')


def main():
    
    # define model
    if cfg.MODEL == 'deeplabv3+':
        net = Deeplabv3plus(class_num=cfg.CLASS_NUM)
    else:
        net = UNetv1(class_num=cfg.CLASS_NUM)
    # load pretrained weights
    checkpoint = torch.load(os.path.join(cfg.LOG_DIR, cfg.FINAL_WEIGHTS))
    net.load_state_dict(checkpoint['state_dict'])
    # use cuda if available
    if torch.cuda.is_available():
        net = net.cuda(device=cfg.DEVICE_LIST[0])
    
    # set to eval 
    net.eval()
    # create output dir
    current_date = str(pd.datetime.now()).split('.')[0]
    save_path = os.path.join(cfg.OUTPUT_DIR, current_date)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if cfg.INFER_MODE == 'eval' and cfg.SRC_IMG.endswith('.csv'):
        test_dataset = LaneSegTestDataset(cfg.SRC_IMG)
        eval_data_loader = DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      drop_last=False)
        dataprocess = tqdm(eval_data_loader)
        # init confusion matrix with all zeros
        confusion_matrix = {"TP": {i: 0 for i in range(8)},
                            "TN": {i: 0 for i in range(8)},
                            "FP": {i: 0 for i in range(8)},
                            "FN": {i: 0 for i in range(8)}}

        for data_item in dataprocess:
            # get batch data
            data_image, data_label, img_path = data_item['image'], \
                                               data_item['label'],\
                                               data_item['img_path']
            if torch.cuda.is_available():
                data_image, data_label = data_image.cuda(device=cfg.DEVICE_LIST[0]), \
                                        data_label.cuda(device=cfg.DEVICE_LIST[0])
            # forward to get output
            data_out = net(data_image)
            data_out = data_out.squeeze(0)  # (8,w1,h1)
            pred = torch.argmax(F.softmax(data_out, dim=0), dim=0)  # (w1,h1)
            # save predict label
            img_name = img_path.split('/')[-1]
            label_output(pred, save_path, img_name)
            # update confusion matrix in eval mode
            confusion_matrix = update_confusion_matrix(pred, data_label, confusion_matrix)
        # compute metric in eval mode
        ious = compute_iou(confusion_matrix)
        m_iou = compute_mean(ious)
        precisions = compute_precision(confusion_matrix)
        m_precision = compute_mean(precisions)
        recalls = compute_recall(confusion_matrix)
        m_recall = compute_mean(recalls)
        # save eval information
        with open(os.path.join(save_path, 'eval_result.txt'), 'w') as f:
            f.write('inference mode: "eval" \n')
            f.write('source csv file: {} \n'.format(cfg.SRC_IMG))
            f.write("mean iou: {:.4f} \n".format(m_iou))
            for i in range(8):
                f.write("class '{}' iou : {:.4f} \n".format(i, ious[i]))
            
            f.write("mean precision: {:.4f} \n".format(m_precision))
            for i in range(8):
                f.write("class '{}' precision : {:.4f} \n".format(i, precisions[i]))

            f.write("mean recall: {:.4f} \n".format(m_recall))
            for i in range(8):
                f.write("class '{}' recall : {:.4f} \n".format(i, recalls[i]))
    
    elif cfg.INFER_MODE == 'test':
        csv_name = 'test_mode_{}.csv'.format(current_date)
        get_img_list(csv_name)
        test_dataset = LaneSegTestDataset(csv_name)
        eval_data_loader = DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      drop_last=False)
        dataprocess = tqdm(eval_data_loader)
        
        for data_item in dataprocess:
            # get batch data
            data_image, img_path = data_item['image'], data_item['img_path']
            if torch.cuda.is_available():
                data_image = data_image.cuda(device=cfg.DEVICE_LIST[0])
            # forward to get output
            data_out = net(data_image)
            data_out = data_out.squeeze(0)  # (8,w1,h1)
            pred = torch.argmax(F.softmax(data_out, dim=0), dim=0)  # (w1,h1)
            # save predict label
            img_name = img_path.split('/')[-1]
            label_output(pred, save_path, img_name)
        with open(os.path.join(save_path, 'test_result.txt'), 'w') as f:
            f.write('inference mode: "test" \n')
            f.write('source csv file: {} \n'.format(csv_name))
    else:
        raise ValueError('Please check the inference config parameters:"INFER_MODE" and "SRC_IMG"')


if __name__ == '__main__':
    main()
