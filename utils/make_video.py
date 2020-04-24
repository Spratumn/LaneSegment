import os
import numpy as np
import pandas as pd
import cv2 as cv

from config import Config


cfg = Config()


def get_img_list(flag='color_label'):
    img_dir = os.path.join(cfg.IMAGE_DIR, flag)
    camera_5_imgs = []
    camera_6_imgs = []
    src_dir = pd.read_csv(cfg.SRC_CSV_DIR, header=None, names=["image"])
    src_dir = src_dir["image"].values[1:]
    index_count = 0
    for img_name in os.listdir(img_dir):
        src_img_dir = src_dir[index_count].split('/')[-1]
        if img_name.endswith('5_bin.png') and img_name[:16] == src_img_dir[:16]:
            camera_5_imgs.append((os.path.join(img_dir, img_name), src_dir[index_count]))
        elif img_name.endswith('6_bin.png') and img_name[:16] == src_img_dir[:16]:
            camera_6_imgs.append((os.path.join(img_dir, img_name), src_dir[index_count]))
        else:
            print(img_name, src_img_dir)
    index_count += 1
    return camera_5_imgs, camera_6_imgs


if __name__ == '__main__':
    camera_5, camera_6 = get_img_list()
    # output camera_5
    video_dir = os.path.join(cfg.IMAGE_DIR, 'camera_5.avi')
    size = (3384, 3420)  # (w,h)
    fps = 5
    videowriter = cv.VideoWriter(video_dir, cv.VideoWriter_fourcc(*'XVID'), fps, size, True)
    for image_path in camera_5:
        label_dir, src_dir = image_path
        label = cv.imread(label_dir)
        src_img = cv.imread(src_dir)

        out_img = np.concatenate((src_img, label), axis=0)
        videowriter.write(out_img)
    videowriter.release()


