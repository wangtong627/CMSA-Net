# # Test code.

# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# import sys

# # Add the target path to the Python search path.
# target_path = "/path/to/CMSA-Net/cmsa_project"
# if target_path not in sys.path:
#     sys.path.append(target_path)

# #

import os
import sys
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Resize

current_path = os.path.abspath(__file__)
sys_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
sys.path.append(sys_path)

from scripts.config import config
from lib.dataloader.preprocess import *


class VideoDataset(Dataset):
    def __init__(self, video_dataset, transform=None, time_interval=1, ttype="train", img_only=False):
        super(VideoDataset, self).__init__()
        # 5 frames for one batch
        self.time_clips = config.video_time_clips  # time_clip 5
        self.video_train_list = []
        self.ttype = ttype  # train
        self.img_only = img_only  # Load images only during testing.

        video_root = os.path.join(config.dataset_root, video_dataset)
        img_root = os.path.join(video_root, "Frame")
        gt_root = os.path.join(video_root, "GT")

        cls_list = os.listdir(img_root)  # Case folders.
        self.video_filelist = {}  # File list grouped by video: {case_name: [(img_path, gt_path), ...]}.
        # cls is the case id.
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)

            tmp_list = os.listdir(cls_img_path)  # Frames in the current case folder.

            # SUN-SEG dataset sorting.
            if "SUN-SEG" in config.dataset_root:
                tmp_list = list(filter(lambda x: "case" in x, tmp_list))
                if "train" in ttype:
                    tmp_list.sort(
                        key=lambda name: (
                            int(name.split("-")[0].split("_")[-1]),  # id 1
                            int(name.split("_a")[1].split("_")[0]),  # id 2
                            int(name.split("_image")[1].split(".jpg")[0]),  # id 3
                        )
                    )
                else:
                    tmp_list.sort(key=lambda name: (int(name.split("_a")[1].split("_")[0]), int(name.split("_image")[1].split(".jpg")[0])))

            # ClinicDB dataset sorting.
            elif "ClinicDB" in config.dataset_root:
                tmp_list.sort(key=lambda name: (int(name.split(".")[0])))

            # Add to video_filelist.
            for filename in tmp_list:
                self.video_filelist[cls].append(
                    (os.path.join(cls_img_path, filename), os.path.join(cls_label_path, filename.replace(".jpg", ".png")))
                )

        # ensemble
        # Iterate over video clips.
        for cls in cls_list:
            li = self.video_filelist[cls]  # All frames in the current video clip.

            # Training mode: build overlapping video clips.
            if "train" in ttype:
                # time_interval = 1
                for begin in range(1, len(li) - (self.time_clips - 1) * time_interval - 1):
                    batch_clips = []
                    # 1. VOS-style input: always use frame 0 as the reference.
                    batch_clips.append(li[0])
                    for t in range(self.time_clips):
                        # 2. Add consecutive time_clips frames.
                        batch_clips.append(li[begin + time_interval * t])
                    # 3. Add to the training list.
                    self.video_train_list.append(batch_clips)

            # Validation/test mode: non-overlapping windows.
            else:
                begin = 0  # change for inference from first frame
                while begin < len(li):
                    # Shift the window if the remaining frames are not enough.
                    if len(li) - begin - 1 < self.time_clips:
                        begin = len(li) - self.time_clips
                    batch_clips = []
                    batch_clips.append(li[0])  # Reference frame.
                    for t in range(self.time_clips):
                        batch_clips.append(li[begin + time_interval * t])
                    begin += self.time_clips  # Non-overlapping window length.
                    self.video_train_list.append(batch_clips)

        self.img_label_transform = transform

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        for _, (img_path, label_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert("RGB")
            if not self.img_only:  # Training uses PIL for data augmentation.
                label = Image.open(label_path).convert("L")
            else:  # Testing does not need augmentation; OpenCV is faster.
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            img_li.append(img)
            label_li.append(label)
        img_li, label_li = self.img_label_transform(img_li, label_li)

        for frame_idx, (img, label) in enumerate(zip(img_li, label_li)):
            if frame_idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                if not self.img_only:
                    # LABEL = torch.zeros(len(img_li) - 1, *(label.shape))
                    LABEL = torch.zeros(len(img_li), *(label.shape))
                else:
                    # LABEL = np.zeros((len(img_li) - 1, 1, *(label.shape)), dtype=np.uint8)
                    LABEL = np.zeros((len(img_li), 1, *(label.shape)), dtype=np.uint8)
                IMG[frame_idx, :, :, :] = img
            else:
                IMG[frame_idx, :, :, :] = img
                # LABEL[frame_idx - 1, :, :, :] = label
            LABEL[frame_idx, :, :, :] = label
        return IMG, LABEL, img_label_li

    def __len__(self):
        return len(self.video_train_list)


class Test_Dataset(Dataset):
    def __init__(self, root, testset):
        time_interval = 1

        self.time_clips = config.video_time_clips  # 5
        self.video_test_list = []

        video_root = os.path.join(root, testset, "Frame")
        cls_list = os.listdir(video_root)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []
            cls_path = os.path.join(video_root, cls)
            tmp_list = os.listdir(cls_path)
            tmp_list = list(filter(lambda x: "case" in x, tmp_list))

            tmp_list.sort(
                key=lambda name: (
                    # int(name.split('-')[0].split('_')[-1]),
                    int(name.split("_a")[1].split("_")[0]),
                    int(name.split("_image")[1].split(".jpg")[0]),
                )
            )

            for filename in tmp_list:
                self.video_filelist[cls].append(os.path.join(cls_path, filename))

        # ensemble
        for cls in cls_list:
            li = self.video_filelist[cls]
            begin = 0  # change for inference from first frame
            while begin < len(li):
                if len(li) - begin - 1 < self.time_clips:
                    begin = len(li) - self.time_clips
                batch_clips = []
                batch_clips.append(li[0])
                for t in range(self.time_clips):
                    batch_clips.append(li[begin + time_interval * t])
                begin += self.time_clips
                self.video_test_list.append(batch_clips)

        """
        In statistics:
            'mean': array([0.4732661 , 0.44874457, 0.3948762]
            'std': array([0.22674961, 0.22012031, 0.2238305]
        """
        self.img_transform = Compose(
            [
                Resize((config.size[0], config.size[1]), Image.BILINEAR),
                ToTensor(),
                Normalize([0.4732661, 0.44874457, 0.3948762], [0.22674961, 0.22012031, 0.2238305]),
            ]
        )

    def __getitem__(self, sample_idx):
        img_path_li = self.video_test_list[sample_idx]
        IMG = None
        img_li = []

        for _, img_path in enumerate(img_path_li):
            img = Image.open(img_path).convert("RGB")
            img_li.append(self.img_transform(img))

        for frame_idx, img in enumerate(img_li):
            if IMG is not None:
                IMG[frame_idx, :, :, :] = img
            else:
                IMG = torch.zeros(len(img_li), *(img.shape))
                IMG[frame_idx, :, :, :] = img
        return IMG, img_path_li

    def __len__(self):
        return len(self.video_test_list)


def get_video_dataset(dataset_name=None, ttype="train"):
    """
    In statistics:
        'mean': array([0.4732661 , 0.44874457, 0.3948762]
        'std': array([0.22674961, 0.22012031, 0.2238305]
    """
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel(
        [
            Resize_video(config.size[0], config.size[1]),
            Random_crop_Resize_Video(20),
            Random_horizontal_flip_video(0.5),
            Random_rotation(0.5, angle=25),
            toTensor_video(),
            Normalize_video(statistics["mean"], statistics["std"]),
        ]
    )
    tf_img_only = config.tf_img_only
    trsf_eval = Compose_imglabel(
        [
            Resize_video(config.size[0], config.size[1], img_only=tf_img_only),
            toTensor_video(img_only=tf_img_only),
            Normalize_video(statistics["mean"], statistics["std"]),
        ]
    )

    if "train" in ttype:
        train_loader = VideoDataset(dataset_name, transform=trsf_main, time_interval=1, ttype=ttype)
    else:
        train_loader = VideoDataset(dataset_name, transform=trsf_eval, time_interval=1, ttype=ttype, img_only=tf_img_only)

    return train_loader



class CMSAVideoDataset(Dataset):
    """
    Test-time sliding window version matching the training window construction.
    Predict one frame at a time without pre-expanding the window list.
    Uses two reference frames: frame 0 and frame 1.
    """

    def __init__(self, video_dataset, transform=None, time_interval=1, ttype="train", img_only=False):
        super(CMSAVideoDataset, self).__init__()
        # 5 frames for one batch
        self.time_clips = config.video_time_clips  # Read from config.
        self.video_train_list = []
        self.ttype = ttype  # train
        self.img_only = img_only  # Load images only during testing.

        video_root = os.path.join(config.dataset_root, video_dataset)  # Full dataset path.
        img_root = os.path.join(video_root, "Frame")
        gt_root = os.path.join(video_root, "GT")

        cls_list = os.listdir(img_root)  # Case folders.
        self.video_filelist = {}  # File list grouped by video: {case_name: [(img_path, gt_path), ...]}.
        # cls is the case id.
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)

            tmp_list = os.listdir(cls_img_path)  # Frames in the current case folder.

            # SUN-SEG dataset sorting.
            if "SUN-SEG" in config.dataset_root:
                tmp_list = list(filter(lambda x: "case" in x, tmp_list))
                if "train" in ttype:
                    tmp_list.sort(
                        key=lambda name: (
                            int(name.split("-")[0].split("_")[-1]),  # id 1
                            int(name.split("_a")[1].split("_")[0]),  # id 2
                            int(name.split("_image")[1].split(".jpg")[0]),  # id 3
                        )
                    )
                else:
                    tmp_list.sort(key=lambda name: (int(name.split("_a")[1].split("_")[0]), int(name.split("_image")[1].split(".jpg")[0])))

            # ClinicDB dataset sorting.
            elif "ClinicDB" in config.dataset_root:
                tmp_list.sort(key=lambda name: (int(name.split(".")[0])))

            # Add to video_filelist.
            for filename in tmp_list:
                self.video_filelist[cls].append(
                    (os.path.join(cls_img_path, filename), os.path.join(cls_label_path, filename.replace(".jpg", ".png")))
                )

        # ensemble
        # Iterate over video clips.
        for cls in cls_list:
            li = self.video_filelist[cls]  # All frames in the current video clip.

            # Training mode: build overlapping video clips.
            if "train" in ttype:
                # time_interval = 1
                for begin in range(1, len(li) - (self.time_clips - 1) * time_interval - 1):
                    batch_clips = []
                    # Use two fixed reference frames.
                    batch_clips.append(li[0])  # f0
                    batch_clips.append(li[1])  # f1
                    for t in range(self.time_clips):
                        # 2. Add consecutive time_clips frames.
                        batch_clips.append(li[begin + time_interval * t])
                    # 3. Add to the training list.
                    self.video_train_list.append(batch_clips)

            # Validation/test mode: slide by one frame.
            else:
                # begin = 0  # change for inference from first frame
                # while begin < len(li):
                #     # Shift the window if the remaining frames are not enough.
                #     if len(li) - begin - 1 < self.time_clips:
                #         begin = len(li) - self.time_clips
                #     batch_clips = []
                #     batch_clips.append(li[0])  # Reference frame.
                #     for t in range(self.time_clips):
                #         batch_clips.append(li[begin + time_interval * t])
                #     # begin += self.time_clips  # Non-overlapping window length.
                #     begin += 1
                #     self.video_train_list.append(batch_clips)
                max_begin = len(li) - self.time_clips
                for begin in range(0, max_begin + 1):

                    # batch_clips = [li[0]]
                    batch_clips = []

                    # Use two fixed references.
                    batch_clips.append(li[0])  # f0
                    batch_clips.append(li[1])  # f1

                    for t in range(self.time_clips):
                        batch_clips.append(li[begin + t])
                    self.video_train_list.append(batch_clips)

        self.img_label_transform = transform

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        for _, (img_path, label_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert("RGB")
            if not self.img_only:  # Training uses PIL for data augmentation.
                label = Image.open(label_path).convert("L")
            else:  # Testing does not need augmentation; OpenCV is faster.
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            img_li.append(img)
            label_li.append(label)
        img_li, label_li = self.img_label_transform(img_li, label_li)

        for frame_idx, (img, label) in enumerate(zip(img_li, label_li)):
            if frame_idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                if not self.img_only:
                    # LABEL = torch.zeros(len(img_li) - 1, *(label.shape))
                    LABEL = torch.zeros(len(img_li), *(label.shape))
                else:
                    # LABEL = np.zeros((len(img_li) - 1, 1, *(label.shape)), dtype=np.uint8)
                    LABEL = np.zeros((len(img_li), 1, *(label.shape)), dtype=np.uint8)
                IMG[frame_idx, :, :, :] = img
            else:
                IMG[frame_idx, :, :, :] = img
                # LABEL[frame_idx - 1, :, :, :] = label
            LABEL[frame_idx, :, :, :] = label
        return IMG, LABEL, img_label_li

    def __len__(self):
        return len(self.video_train_list)


def get_cmsa_video_dataset(dataset_name=None, ttype="train"):
    """
    In statistics:
        'mean': array([0.4732661 , 0.44874457, 0.3948762]
        'std': array([0.22674961, 0.22012031, 0.2238305]
    """
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel(
        [
            Resize_video(config.size[0], config.size[1]),
            Random_crop_Resize_Video(20),
            Random_horizontal_flip_video(0.5),
            Random_rotation(0.5, angle=25),
            toTensor_video(),
            Normalize_video(statistics["mean"], statistics["std"]),
        ]
    )
    tf_img_only = config.tf_img_only
    trsf_eval = Compose_imglabel(
        [
            Resize_video(config.size[0], config.size[1], img_only=tf_img_only),
            toTensor_video(img_only=tf_img_only),
            Normalize_video(statistics["mean"], statistics["std"]),
        ]
    )

    if "train" in ttype:
        train_loader = CMSAVideoDataset(dataset_name, transform=trsf_main, time_interval=1, ttype=ttype)
    else:
        train_loader = CMSAVideoDataset(dataset_name, transform=trsf_eval, time_interval=1, ttype=ttype, img_only=tf_img_only)

    return train_loader


