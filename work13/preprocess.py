import os
import json
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import (
    LABEL_NAMES,
    RAW_VIDEO_DIR,
    PROCESSED_DIR,
    TARGET_FRAMES,
    INPUT_DIM,
    ensure_dir,
    label_to_id,
    id_to_label,
)


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def extract_pose_from_frame(frame, pose):
    """
    输入一帧图像，输出一帧骨架特征，形状为 [132]
    MediaPipe Pose 有 33 个点，每个点有 x, y, z, visibility 四个值。
    33 * 4 = 132
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks is None:
        return np.zeros(INPUT_DIM, dtype=np.float32)

    features = []
    for landmark in result.pose_landmarks.landmark:
        features.extend([
            landmark.x,
            landmark.y,
            landmark.z,
            landmark.visibility,
        ])

    return np.array(features, dtype=np.float32)


def normalize_skeleton(frame_feature):
    """
    对一帧骨架做简单归一化：
    1. 以左右髋部中心作为原点
    2. 用肩宽作为尺度
    """
    points = frame_feature.reshape(33, 4).copy()

    # 如果这一帧没有检测到人体，直接返回 0
    if np.all(points == 0):
        return frame_feature

    # MediaPipe Pose 索引：
    # 11 = left shoulder, 12 = right shoulder
    # 23 = left hip, 24 = right hip
    left_shoulder = points[11, :3]
    right_shoulder = points[12, :3]
    left_hip = points[23, :3]
    right_hip = points[24, :3]

    hip_center = (left_hip + right_hip) / 2.0
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    # 只归一化 x, y, z，不改变 visibility
    points[:, :3] = (points[:, :3] - hip_center) / shoulder_width

    return points.reshape(-1).astype(np.float32)


def resample_sequence(sequence, target_frames=TARGET_FRAMES):
    """
    把不同长度的视频骨架序列统一成 target_frames 帧。
    输入 shape: [原始帧数, 132]
    输出 shape: [target_frames, 132]
    """
    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) == 0:
        return np.zeros((target_frames, INPUT_DIM), dtype=np.float32)

    if len(sequence) == target_frames:
        return sequence

    indices = np.linspace(0, len(sequence) - 1, target_frames)
    indices = np.round(indices).astype(int)

    return sequence[indices]


def video_to_sequence(video_path, target_frames=TARGET_FRAMES):
    """
    把单个视频转换为骨架序列 [target_frames, 132]
    """
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None

    sequence = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            feature = extract_pose_from_frame(frame, pose)
            feature = normalize_skeleton(feature)
            sequence.append(feature)

    cap.release()

    sequence = resample_sequence(sequence, target_frames)
    return sequence


def collect_video_paths():
    """
    遍历 data/raw_videos 下的所有类别文件夹，收集视频路径和标签。
    """
    data = []
    label_dict = label_to_id()

    for class_name in LABEL_NAMES:
        class_dir = os.path.join(RAW_VIDEO_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"警告：找不到类别文件夹: {class_dir}")
            continue

        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(VIDEO_EXTENSIONS):
                video_path = os.path.join(class_dir, file_name)
                label = label_dict[class_name]
                data.append((video_path, label))

    return data


def main():
    ensure_dir(PROCESSED_DIR)

    all_videos = collect_video_paths()

    if len(all_videos) == 0:
        print("没有找到任何视频。请检查 data/raw_videos 下面是否放了视频。")
        return

    print(f"共找到 {len(all_videos)} 个视频。开始提取骨架...")

    X = []
    y = []

    for video_path, label in tqdm(all_videos):
        sequence = video_to_sequence(video_path, TARGET_FRAMES)

        if sequence is None:
            continue

        X.append(sequence)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print("骨架数据形状 X:", X.shape)
    print("标签数据形状 y:", y.shape)

    # 如果每个类别至少有 2 个样本，就使用 stratify 保持类别比例
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = np.all(counts >= 2)

    if can_stratify:
        stratify = y
    else:
        print("警告：某些类别样本太少，无法按类别比例划分。将使用普通随机划分。")
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    label_map = {
        "label_to_id": label_to_id(),
        "id_to_label": id_to_label(),
    }

    with open(os.path.join(PROCESSED_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)

    print("预处理完成。文件已保存到 data/processed/")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)


if __name__ == "__main__":
    main()