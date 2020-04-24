from tqdm import tqdm
import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from model.deeplabv3p import Deeplabv3plus
from model.unet import UNetv1, UNetv2
from utils.image_process import LaneSegTrainDataset, ToTensor, ImageAug
from utils.loss import CrossEntropyLoss, FocalLoss
from utils.metric import update_confusion_matrix, compute_iou, \
    compute_precision, compute_recall, compute_mean

from config import Config

cfg = Config()


def train_epoch(net, epoch, data_loader, optimizer, criterion, train_log):
    # set to train 
    net.train()
    total_loss = 0.0
    data_process = tqdm(data_loader)

    for batch_item in data_process:
        # get batch data
        batch_image, batch_label = batch_item['image'], batch_item['label']
        if torch.cuda.is_available():
            if cfg.MULTI_GPU:
                batch_image, batch_label = batch_image.cuda(device=cfg.DEVICE_LIST[0]), \
                                           batch_label.cuda(device=cfg.DEVICE_LIST[0])
            else:
                batch_image, batch_label = batch_image.cuda(), \
                                           batch_label.cuda()
        # zero the gradient buffers
        optimizer.zero_grad()
        # forward to get output
        batch_out = net(batch_image)
        # compute loss
        batch_loss = criterion(batch_out, batch_label)
        total_loss += batch_loss.item()
        # auto backward
        batch_loss.backward()
        # update weights
        optimizer.step()
        # print batch result
        data_process.set_description_str("epoch:{}".format(epoch))
        data_process.set_postfix_str("batch_loss:{:.4f}".format(batch_loss.item()))
    # when epoch finished write to log file
    # log_title = 'epoch,average loss'
    log_string = "{},{:.4f} \n".format(epoch, total_loss / len(data_loader))
    train_log.write(log_string)
    train_log.flush()


def eval_epoch(net, epoch, data_loader, eval_log):
    # set to eval 
    net.eval()
    total_loss = 0.0
    data_process = tqdm(data_loader)
    # init confusion matrix with all zeros
    confusion_matrix = {"TP": {i: 0 for i in range(8)},
                        "TN": {i: 0 for i in range(8)},
                        "FP": {i: 0 for i in range(8)},
                        "FN": {i: 0 for i in range(8)}}

    for batch_item in data_process:
        # get batch data
        batch_image, batch_label = batch_item['image'], batch_item['label']
        if cfg.MULTI_GPU:
            batch_image, batch_label = batch_image.cuda(device=cfg.DEVICE_LIST[0]), \
                                       batch_label.cuda(device=cfg.DEVICE_LIST[0])
        else:
            batch_image, batch_label = batch_image.cuda(), \
                                       batch_label.cuda()
        # forward to get output
        batch_out = net(batch_image)
        # compute loss
        # batch_loss = CrossEntropyLoss(cfg.CLASS_NUM)(batch_out, batch_label)
        batch_loss = FocalLoss(cfg.CLASS_NUM)(batch_out, batch_label)
        total_loss += batch_loss.detach().item()

        # get prediction, shape and value type same as batch_label
        pred = torch.argmax(F.softmax(batch_out, dim=1), dim=1)
        # compute confusion matrix using batch data
        confusion_matrix = update_confusion_matrix(pred, batch_label, confusion_matrix)
        # print batch result
        data_process.set_description_str("epoch:{}".format(epoch))
        data_process.set_postfix_str("batch_loss:{:.4f}".format(batch_loss))

    eval_loss = total_loss / len(data_loader)

    # compute metric
    epoch_ious = compute_iou(confusion_matrix)
    epoch_m_iou = compute_mean(epoch_ious)
    epoch_precisions = compute_precision(confusion_matrix)
    epoch_m_precision = compute_mean(epoch_precisions)
    epoch_recalls = compute_recall(confusion_matrix)
    epoch_m_recall = compute_mean(epoch_recalls)

    # print eval iou every epoch
    print('mean iou: {} \n'.format(epoch_m_iou))
    for i in range(8):
        print_string = "class '{}' iou : {:.4f} \n".format(i, epoch_ious[i])
        print(print_string)

    # make log string
    log_values = [epoch, eval_loss, epoch_m_iou] + \
                 epoch_ious + [epoch_m_precision] + \
                 epoch_precisions + [epoch_m_recall] + epoch_recalls
    log_values = [str(v) for v in log_values]
    log_string = ','.join(log_values)
    eval_log.write(log_string + '\n')
    eval_log.flush()


def main():

    # using multi process to load data when cuda is available
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    # set image augment
    augments = [ImageAug(), ToTensor()]
    # get dataset and iterable dataloader
    train_dataset = LaneSegTrainDataset("train.csv",
                                        transform=transforms.Compose(augments))
    eval_dataset = LaneSegTrainDataset("eval.csv",
                                       transform=transforms.Compose([ToTensor()]))
    if cfg.MULTI_GPU:
        train_batch_size = cfg.TRAIN_BATCH_SIZE * len(cfg.DEVICE_LIST)
        eval_batch_size = cfg.EVAL_BATCH_SIZE * len(cfg.DEVICE_LIST)

    else:
        train_batch_size = cfg.TRAIN_BATCH_SIZE
        eval_batch_size = cfg.EVAL_BATCH_SIZE
    train_data_batch = DataLoader(train_dataset,
                                  batch_size=train_batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  **kwargs)
    eval_data_batch = DataLoader(eval_dataset,
                                 batch_size=eval_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 **kwargs)

    # define model
    if cfg.MODEL == 'deeplabv3+':
        net = Deeplabv3plus(class_num=cfg.CLASS_NUM, normal=cfg.NORMAL)
    elif cfg.MODEL == 'unet':
        net = UNetv1(class_num=cfg.CLASS_NUM, normal=cfg.NORMAL)
    else:
        net = UNetv1(class_num=cfg.CLASS_NUM, normal=cfg.NORMAL)

    # use cuda if available
    if torch.cuda.is_available():
        if cfg.MULTI_GPU:
            net = torch.nn.DataParallel(net, device_ids=cfg.DEVICE_LIST)
            net = net.cuda(device=cfg.DEVICE_LIST[0])
        else:
            net = net.cuda()
        # load pretrained weights
        if cfg.PRE_TRAINED:
            checkpoint = torch.load(os.path.join(cfg.LOG_DIR, cfg.PRE_TRAIN_WEIGHTS))
            net.load_state_dict(checkpoint['state_dict'])

    # define optimizer
    # optimizer = torch.optim.SGD(net.parameters(),
    #                             lr=cfg.BASE_LR,
    #                             momentum=0.9,
    #                             weight_decay=cfg.WEIGHT_DECAY)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=cfg.BASE_LR,
                                 weight_decay=cfg.WEIGHT_DECAY)
    # criterion = CrossEntropyLoss(cfg.CLASS_NUM)
    criterion = FocalLoss(cfg.CLASS_NUM)
    # define log file
    train_log = open(os.path.join(cfg.LOG_DIR, "train_log_{}.csv".format(cfg.TRAIN_NUMBER)), 'w')
    train_log_title = "epoch,average loss\n"
    train_log.write(train_log_title)
    train_log.flush()
    eval_log = open(os.path.join(cfg.LOG_DIR, "eval_log_{}.csv".format(cfg.TRAIN_NUMBER)), 'w')
    eval_log_title = "epoch,average_loss, mean_iou, iou_0,iou_1,iou_2,iou_3,iou_4,iou_5,iou_6,iou_7, " \
                     "mean_precision, precision_0,precision_1,precision_2,precision_3, precision_4," \
                     "precision_5,precision_6,precision_7, mean_recall, recall_0,recall_1,recall_2," \
                     "recall_3,recall_4,recall_5,recall_6,recall_7\n"
    eval_log.write(eval_log_title)
    eval_log.flush()

    # train and test epoch by epoch
    for epoch in range(cfg.EPOCHS):
        print('current epoch learning rate: {}'.format(cfg.BASE_LR))
        train_epoch(net, epoch, train_data_batch, optimizer, criterion, train_log)
        # save model
        if epoch != cfg.EPOCHS - 1:
            torch.save({'state_dict': net.state_dict()},
                       os.path.join(cfg.LOG_DIR,
                                    "laneNet_{0}_{1}th_epoch_{2}.pt".format(cfg.MODEL,
                                                                            cfg.TRAIN_NUMBER,
                                                                            epoch)))
        else:
            torch.save({'state_dict': net.state_dict()},
                       os.path.join(cfg.LOG_DIR,
                                    "laneNet_{0}_{1}th.pt".format(cfg.MODEL, cfg.TRAIN_NUMBER)))
        eval_epoch(net, epoch, eval_data_batch, eval_log)

    train_log.close()
    eval_log.close()


if __name__ == '__main__':
    main()
