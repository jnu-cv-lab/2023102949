import argparse
import torch
import torch.nn.functional as F

from model import SkeletonTransformer
from preprocess import video_to_sequence
from utils import INPUT_DIM, TARGET_FRAMES, NUM_CLASSES, LABEL_NAMES, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="要预测的视频路径")
    parser.add_argument("--model_path", type=str, default="runs/best_model.pth")
    args = parser.parse_args()

    device = get_device()
    print("当前使用设备:", device)

    checkpoint = torch.load(args.model_path, map_location=device)

    label_names = checkpoint.get("label_names", LABEL_NAMES)
    target_frames = checkpoint.get("target_frames", TARGET_FRAMES)

    model = SkeletonTransformer(
        input_dim=INPUT_DIM,
        target_frames=target_frames,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        num_classes=NUM_CLASSES,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sequence = video_to_sequence(args.video, target_frames=target_frames)

    if sequence is None:
        print("视频读取失败，无法推理。")
        return

    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        confidence, pred_id = torch.max(probs, dim=1)

    pred_id = pred_id.item()
    confidence = confidence.item()

    print("\n========== 推理结果 ==========")
    print(f"Predicted class: {label_names[pred_id]}")
    print(f"Confidence: {confidence:.4f}")

    print("\n所有类别概率:")
    for i, class_name in enumerate(label_names):
        print(f"{class_name}: {probs[0, i].item():.4f}")


if __name__ == "__main__":
    main()