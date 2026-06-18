import os
import random
import numpy as np
import torch


# 按照作业要求固定 6 个类别
# 注意：这里的名字必须和 data/raw_videos 里面的文件夹名字一致
LABEL_NAMES = [
    "forehand_drive",
    "forehand_lift",
    "forehand_net_shot",
    "forehand_clear",
    "backhand_drive",
    "backhand_net_shot",
]

NUM_CLASSES = len(LABEL_NAMES)

RAW_VIDEO_DIR = "data/raw_videos"
PROCESSED_DIR = "data/processed"

TARGET_FRAMES = 30
INPUT_DIM = 132


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def label_to_id():
    return {name: idx for idx, name in enumerate(LABEL_NAMES)}


def id_to_label():
    return {idx: name for idx, name in enumerate(LABEL_NAMES)}