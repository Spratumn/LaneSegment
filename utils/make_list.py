import os
import pandas as pd  
from sklearn.utils import shuffle  # random


"""
将训练集中的原始图片与label图片的路径进行对应组合，并打乱顺序保存到csv
"""

CSV_DIR = "../data_list"
IMAGE_DIR = r"D:\Data\Open\DataSet\BaiduAI\train_set\Image_Data"
LABEL_DIR = r"D:\Data\Open\DataSet\BaiduAI\train_set\Labels\Labels_Fixed"

DATA_PARTITION = [0.8, 0.9]


def get_data_dir():

    image_list = []
    label_list = []
    
    for s1 in os.listdir(IMAGE_DIR):
        image_sub_dir1 = os.path.join(IMAGE_DIR, s1)
        label_sub_dir1 = os.path.join(LABEL_DIR, 'Label_' + str.lower(s1), 'Label')
        for s2 in os.listdir(image_sub_dir1):
            image_sub_dir2 = os.path.join(image_sub_dir1, s2)
            label_sub_dir2 = os.path.join(label_sub_dir1, s2)
            for s3 in os.listdir(image_sub_dir2):
                image_sub_dir3 = os.path.join(image_sub_dir2, s3)
                label_sub_dir3 = os.path.join(label_sub_dir2, s3)
                for image_name in os.listdir(image_sub_dir3):
                    label_name = image_name.replace('.jpg', '_bin.png')
                    image_sub_dir4 = os.path.join(image_sub_dir3, image_name)
                    label_sub_dir4 = os.path.join(label_sub_dir3, label_name)                                 
                    if not os.path.exists(label_sub_dir4):
                        print(image_sub_dir4)
                        print(label_sub_dir4)
                    else:
                        image_list.append(image_sub_dir4)
                        label_list.append(label_sub_dir4)
    assert len(image_list) == len(label_list)
    return image_list, label_list


def make_datalist():
    image_list, label_list = get_data_dir()
    # set partition size
    total_length = len(image_list)
    eval_part = int(total_length*DATA_PARTITION[0])
    test_part = int(total_length*DATA_PARTITION[1])
    # 
    all_data = pd.DataFrame({'image': image_list, 'label': label_list})
    all_shuffle = shuffle(all_data)

    train_dataset = all_shuffle[:eval_part]
    eval_dataset = all_shuffle[eval_part:test_part]
    test_dataset = all_shuffle[test_part:]

    little_train_dataset = train_dataset[:20]
    little_eval_dataset = eval_dataset[:10]
    little_test_dataset = test_dataset[:10]

    train_dataset.to_csv(CSV_DIR + "/train.csv", index=False)
    eval_dataset.to_csv(CSV_DIR + "/eval.csv", index=False)
    test_dataset.to_csv(CSV_DIR + "/test.csv", index=False)

    little_train_dataset.to_csv(CSV_DIR + "/little_train.csv", index=False)
    little_eval_dataset.to_csv(CSV_DIR + "/little_eval.csv", index=False)
    little_test_dataset.to_csv(CSV_DIR + "/little_test.csv", index=False)


if __name__ == '__main__':
    make_datalist()






