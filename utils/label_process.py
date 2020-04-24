import numpy as np
from PIL import Image


def label_encoder(mark_label):
    """mark label to train label"""

    train_label = np.zeros((mark_label.shape[0], mark_label.shape[1]))

    id_train = {0: [0, 249, 255, 213, 206, 207, 211, 208, 216, 215, 218,
                    219, 232, 202, 231, 230, 228, 229, 233, 212, 223],
                1: [200, 204, 209],
                2: [201, 203],
                3: [217],
                4: [210],
                5: [214],
                6: [220, 221, 222, 224, 225, 226],
                7: [205, 227, 250]}
    for i in range(8):
        for item in id_train[i]:
            train_label[mark_label == item] = i

    return train_label


def label_id_decoder(train_label):
    """train label to mark label for id"""
    mark_label = np.zeros((train_label.shape[0], train_label.shape[1]), dtype='uint8')
    # 0
    mark_label[train_label == 0] = 0
    # 1
    mark_label[train_label == 1] = 204
    # 2 
    mark_label[train_label == 2] = 203
    # 3
    mark_label[train_label == 3] = 217
    # 4
    mark_label[train_label == 4] = 210
    # 5
    mark_label[train_label == 5] = 214
    # 6
    mark_label[train_label == 6] = 224
    # 7
    mark_label[train_label == 7] = 227

    return mark_label


def label_color_decoder(train_label):
    """train label to mark label for color map"""
    mark_label = np.zeros((3, train_label.shape[0], train_label.shape[1]), dtype='uint8')
    # 0
    mark_label[0][train_label == 0] = 0
    mark_label[1][train_label == 0] = 0
    mark_label[2][train_label == 0] = 0
    # 1
    mark_label[0][train_label == 1] = 70
    mark_label[1][train_label == 1] = 130
    mark_label[2][train_label == 1] = 180
    # 2
    mark_label[0][train_label == 2] = 0
    mark_label[1][train_label == 2] = 0
    mark_label[2][train_label == 2] = 142
    # 3
    mark_label[0][train_label == 3] = 153
    mark_label[1][train_label == 3] = 153
    mark_label[2][train_label == 3] = 153
    # 4
    mark_label[0][train_label == 4] = 128
    mark_label[1][train_label == 4] = 64
    mark_label[2][train_label == 4] = 128
    # 5
    mark_label[0][train_label == 5] = 190
    mark_label[1][train_label == 5] = 153
    mark_label[2][train_label == 5] = 153
    # 6
    mark_label[0][train_label == 6] = 0
    mark_label[1][train_label == 6] = 0
    mark_label[2][train_label == 6] = 230
    # 7
    mark_label[0][train_label == 7] = 255
    mark_label[1][train_label == 7] = 128
    mark_label[2][train_label == 7] = 0

    return mark_label


def get_id(train_label):
    pixels = [0.0]
    for h in range(train_label.shape[0]):
        for w in range(train_label.shape[1]):
            pixel = train_label[h, w]
            if pixel not in pixels:
                pixels.append(pixel)
    return pixels




