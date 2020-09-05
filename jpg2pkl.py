import os
import numpy as np
import cv2
import sys
import glob
import pickle
from multiprocessing import Pool


label_dic = np.load('work/UCF-101_jpg/label_dir.npy', allow_pickle=True).item()
print(label_dic)

source_dir = 'work/UCF-101_jpg'
target_train_dir = 'work/train'
target_test_dir = 'work/test'
target_val_dir = 'work/val'
if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)
if not os.path.exists(target_val_dir):
    os.mkdir(target_val_dir)
    
for key in label_dic:
    each_mulu = key + '_jpg'  # PommelHorse_jpg

    label_dir = os.path.join(source_dir, each_mulu)  # work/UCF-101_jpg/PommelHorse_jpg
    label_mulu = os.listdir(label_dir)
    video_id = 0
    train_num = int(len(label_mulu)*0.8)
    test_num = int(len(label_mulu)*0.1)
    test_start = test_num + train_num

    # 遍历当前类中的视频文件夹
    for each_label_mulu in label_mulu:
        image_file = os.listdir(os.path.join(label_dir, each_label_mulu))
        # 对列表排序
        image_file.sort()
        # 同一个视频中得到的帧文件名只有后几位不同(_%d.jpg)，
        # 所以取第一个图片的文件名去掉后6位就是整个视频的帧文件名
        print(image_file[0])
        image_name = image_file[0][:-6]
        image_num = len(image_file)
        frame = []
        vid = image_name
        for i in range(image_num):
            image_path = os.path.join(os.path.join(label_dir, each_label_mulu), image_name + '_' + str(i) + '.jpg')
            frame.append(image_path)

        output_pkl = vid + '.pkl'
        if video_id < train_num:
            output_pkl = os.path.join(target_train_dir, output_pkl)
        elif video_id >= train_num and video_id < test_start:
            output_pkl = os.path.join(target_test_dir, output_pkl)
        else:
            output_pkl = os.path.join(target_val_dir, output_pkl)

        f = open(output_pkl, 'wb')
        # Pickle the list using the highest protocol available.(-1)
        pickle.dump((vid, label_dic[key], frame), f, -1)
        f.close()
        video_id += 1
