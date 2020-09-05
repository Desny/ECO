import os
import numpy as np
import cv2
import random

video_src_src_path = 'work/UCF-101'
video_jpg_path = 'work/UCF-101_jpg'

sample_frames = 32

label_name = os.listdir(video_src_src_path)
label_dir = {}
index = 0
for i in label_name:
    if i.startswith('.'):
        continue
    label_dir[i] = index
    index += 1
    video_src_path = os.path.join(video_src_src_path, i)  # work/UCF-101/BaseballPitch
    # 当前运动类别的文件夹
    video_save_path = os.path.join(video_jpg_path, i) + '_jpg'  # work/UCF-101_jpg/BaseballPitch_jpg
    
    if not os.path.exists(video_jpg_path):
        os.mkdir(video_jpg_path)

    if not os.path.exists(video_save_path):
        os.mkdir(video_save_path)

    videos = os.listdir(video_src_path)
    # 过滤出当前动作类别的avi文件
    videos = filter(lambda x: x.endswith('avi'), videos)

    for each_video in videos:
        each_video_name, _ = each_video.split('.')
        if not os.path.exists(video_save_path + '/' + each_video_name):
            os.mkdir(video_save_path + '/' + each_video_name)
        # 每个视频处理得到的jpg文件夹的全路径，以/结尾
        each_video_save_full_path = os.path.join(video_save_path, each_video_name) + '/'  # work/UCF-101_jpg/BaseballPitch_jpg/v_BaseballPitch_g01_c01/
        # 每个视频源文件的全路径，以.avi结尾
        each_video_full_path = os.path.join(video_src_path, each_video)  # work/UCF-101/BaseballPitch/v_BaseballPitch_g01_c01.avi

        cap = cv2.VideoCapture(each_video_full_path)
        # 不足32帧的视频不要
        total_frames = cap.get(7)
        if total_frames < 32:  # 7 -> CAP_PROP_FRAME_COUNT
            os.rmdir(video_save_path + '/' + each_video_name)
            continue
        
        # 32帧索引列表
        sample_list = []  # 设定索引从0开始
        seg_len = int(total_frames / sample_frames)
        start = random.randrange(0, total_frames%seg_len+seg_len)
        sample_list.append(start)
        start += 1
        for i in range(sample_frames-1):
            sample_list.append(random.randrange(start, start+seg_len))
            start = start + seg_len
        frame_count = 0    # 视频中的第几帧
        i = 0
        success = True
        while success:
            # 如果读取一帧成功，就会返回True, 该帧
            # 可以用success判断是否读到了末尾（False -> 读帧失败==文件结束）
            success, frame = cap.read()

            if success and frame_count==sample_list[i]:
                cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % i, frame)
                i += 1
            if i == len(sample_list):
                break
            frame_count += 1
        cap.release()
    # end for videos
# end for labels
np.save(os.path.join(video_jpg_path, 'label_dir.npy'), label_dir)
print(label_dir)
