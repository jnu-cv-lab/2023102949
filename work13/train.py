import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from model import SkeletonTransformer
from utils import (
    PROCESSED_DIR,
    TARGET_FRAMES,
    INPUT_DIM,
    NUM_CLASSES,
    LABEL_NAMES,
    set_seed,
    get_device,
    ensure_dir,
)


def load_data():
    X_train_path = os.path.join(PROCESSED_DIR, "X_train.npy")
    y_train_path = os.path.join(PROCESSED_DIR, "y_train.npy")
    X_test_path = os.path.join(PROCESSED_DIR, "X_test.npy")
    y_test_path = os.path.join(PROCESSED_DIR, "y_test.npy")

    if not os.path.exists(X_train_path):
        raise FileNotFoundError(
            "找不到预处理后的数据。请先运行: python preprocess.py"
        )

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    return X_train, y_train, X_test, y_test


def create_dataloader(X, y, batch_size=16, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            preds = torch.argmax(logits, dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


def save_training_curve(train_losses, test_losses, train_accs, test_accs, save_dir):
    ensure_dir(save_dir)

    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(train_accs, label="train accuracy")
    plt.plot(test_accs, label="test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()


def save_confusion_matrix(y_true, y_pred, save_dir):
    ensure_dir(save_dir)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABEL_NAMES,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    return cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="runs")
    args = parser.parse_args()

    set_seed(42)

    device = get_device()
    print("当前使用设备:", device)

    X_train, y_train, X_test, y_test = load_data()

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    train_loader = create_dataloader(
        X_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_loader = create_dataloader(
        X_test,
        y_test,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = SkeletonTransformer(
        input_dim=INPUT_DIM,
        target_frames=TARGET_FRAMES,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        num_classes=NUM_CLASSES,
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_test_acc = 0.0

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    ensure_dir(args.save_dir)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, test_acc, y_true, y_pred = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "target_frames": TARGET_FRAMES,
                "input_dim": INPUT_DIM,
                "num_classes": NUM_CLASSES,
                "label_names": LABEL_NAMES,
                "best_test_acc": best_test_acc,
            }

            torch.save(
                checkpoint,
                os.path.join(args.save_dir, "best_model.pth"),
            )

    print("\n训练完成。")
    print(f"最佳测试准确率: {best_test_acc:.4f}")

    # 最后一轮测试结果
    test_loss, test_acc, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\n最终测试准确率:", test_acc)

    cm = save_confusion_matrix(y_true, y_pred, args.save_dir)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    save_training_curve(
        train_losses,
        test_losses,
        train_accs,
        test_accs,
        args.save_dir,
    )

    history = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "best_test_acc": best_test_acc,
    }

    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

    print(f"\n结果文件已保存到: {args.save_dir}")
    print("包括：best_model.pth, loss_curve.png, accuracy_curve.png, confusion_matrix.png")


if __name__ == "__main__":
    main()