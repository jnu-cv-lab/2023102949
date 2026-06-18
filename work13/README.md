# Skeleton Transformer for Badminton Action Recognition

## 1. 实验简介

本实验实现了一个基于 **MediaPipe Pose** 和 **Skeleton Transformer** 的羽毛球击球动作识别系统。实验没有直接使用原始视频像素进行分类，而是先从视频中提取人体姿态关键点，将视频转换为骨架时间序列，然后使用轻量级 Transformer Encoder 对动作类别进行分类。

实验目标包括：

- 使用 OpenCV 读取羽毛球击球视频；
- 使用 MediaPipe Pose 提取每一帧的人体 33 个关键点；
- 将每一帧转换为 `33 × 4 = 132` 维骨架特征；
- 将每个视频统一重采样为固定长度的 `[30, 132]` 骨架序列；
- 使用 Transformer Encoder 完成 6 类羽毛球动作分类；
- 输出训练曲线、测试准确率、混淆矩阵和单视频推理结果。

---

## 2. 数据集说明

本实验使用 Kaggle 数据集 `badminton_storke_video`。数据集包含 6 类羽毛球击球动作视频：

| 标签编号 | 英文类别 | 中文说明 |
|---|---|---|
| 0 | `forehand_drive` | 正手平抽 / 正手驱动球 |
| 1 | `forehand_lift` | 正手挑球 |
| 2 | `forehand_net_shot` | 正手网前球 |
| 3 | `forehand_clear` | 正手高远球 |
| 4 | `backhand_drive` | 反手平抽 / 反手驱动球 |
| 5 | `backhand_net_shot` | 反手网前球 |

数据集在项目中的目录结构如下：

```text
Badminton_Transformer/
├── data/
│   ├── raw_videos/
│   │   ├── backhand_drive/
│   │   ├── backhand_net_shot/
│   │   ├── forehand_clear/
│   │   ├── forehand_drive/
│   │   ├── forehand_lift/
│   │   └── forehand_net_shot/
│   └── processed/
├── preprocess.py
├── model.py
├── train.py
├── inference.py
└── utils.py
```

---

## 3. 实验环境

本实验在 VS Code 中完成，主要环境如下：

```text
Operating System: WSL Ubuntu
Python: 3.12
Deep Learning Framework: PyTorch
Pose Estimation: MediaPipe Pose
Video Processing: OpenCV
Evaluation: scikit-learn
Plotting: matplotlib
```

安装依赖：

```bash
pip install numpy opencv-python mediapipe scikit-learn tqdm matplotlib torch
```

---

## 4. 数据预处理方法

预处理由 `preprocess.py` 完成，主要步骤如下：

1. 遍历 `data/raw_videos/` 下的 6 个类别文件夹；
2. 使用 OpenCV 逐帧读取视频；
3. 使用 MediaPipe Pose 提取每帧人体 33 个关键点；
4. 每个关键点包含 `x, y, z, visibility` 四个特征，因此每帧为 132 维；
5. 将不同长度的视频重采样为固定的 30 帧；
6. 对骨架进行归一化处理，以左右髋部中心作为原点，并用肩宽进行尺度归一化；
7. 按照 8:2 划分训练集和测试集；
8. 保存为 `.npy` 文件。

运行预处理命令：

```bash
python preprocess.py --data_dir data/raw_videos --out_dir data/processed --target_frames 30 --test_size 0.2
```

预处理完成后生成的数据文件如下：

```text
data/processed/
├── X_train.npy
├── y_train.npy
├── X_test.npy
├── y_test.npy
├── label_map.json
└── preprocess_summary.json
```

本次实验得到的数据形状为：

```text
X_train: (668, 30, 132)
y_train: (668,)
X_test:  (168, 30, 132)
y_test:  (168,)
```

这说明训练集包含 668 个视频样本，测试集包含 168 个视频样本。每个视频样本都被转换为长度为 30 的骨架序列，每一帧为 132 维特征。

---

## 5. 模型结构

模型定义在 `model.py` 中，核心模型为 `SkeletonTransformer`。

整体结构如下：

```text
Input Skeleton Sequence: [B, 30, 132]
        ↓
Linear Embedding: 132 → 128
        ↓
Position Embedding
        ↓
Transformer Encoder × 2
        ↓
Mean Pooling over Time Dimension
        ↓
MLP Classifier
        ↓
Output Logits: [B, 6]
```

其中：

| 参数 | 数值 | 说明 |
|---|---:|---|
| `input_dim` | 132 | 每帧骨架特征维度 |
| `target_frames` | 30 | 每个视频统一采样帧数 |
| `d_model` | 128 | Transformer 隐藏层维度 |
| `nhead` | 4 | 多头注意力头数 |
| `num_layers` | 2 | Transformer Encoder 层数 |
| `dim_feedforward` | 256 | 前馈网络中间层维度 |
| `num_classes` | 6 | 分类类别数 |
| `dropout` | 0.1 | Dropout 比例 |

`model.py` 本身不需要单独运行，它负责定义模型结构。训练时，`train.py` 通过下面语句调用模型：

```python
from model import SkeletonTransformer
```

推理时，`inference.py` 也会导入同一个模型结构，并加载训练好的 `best_model.pth` 参数。

---

## 6. 训练方法

训练由 `train.py` 完成。训练过程使用：

- 损失函数：`CrossEntropyLoss`
- 优化器：`Adam`
- 学习率：`1e-3`
- Batch size：`16`
- Epochs：`20`

运行训练命令：

```bash
python train.py --epochs 20 --batch_size 16
```

训练结束后，结果保存在 `runs/` 文件夹中：

```text
runs/
├── best_model.pth
├── loss_curve.png
├── accuracy_curve.png
├── confusion_matrix.png
└── history.json
```

---

## 7. 实验结果

### 7.1 Accuracy 曲线

![Training and Test Accuracy](runs/accuracy_curve.png)

从 Accuracy 曲线可以看到，训练集准确率整体持续上升，最终接近 0.60。测试集准确率在训练前期也有所提升，最高测试准确率达到 0.5179，但后期出现波动并下降，最终测试准确率为 0.4226。

这说明模型能够学习到一部分骨架动作特征，但泛化能力仍然有限。

---

### 7.2 Loss 曲线

![Training and Test Loss](runs/loss_curve.png)

从 Loss 曲线可以看到，训练集 loss 持续下降，从约 1.79 降到约 1.03，说明模型在训练集上逐渐拟合数据。但是测试集 loss 没有持续下降，并且在部分 epoch 出现明显波动。

这种现象说明模型可能存在一定程度的过拟合。模型在训练集上的表现不断变好，但在测试集上的表现没有同步提升。

---

### 7.3 混淆矩阵

![Confusion Matrix](runs/confusion_matrix.png)

混淆矩阵如下：

```text
[[20  3  2  3  1  3]
 [ 4 17  4  4  4  2]
 [ 6  5  6  2  1  1]
 [ 9  4  3  8  0  0]
 [ 4  4  4  5  4  1]
 [ 8  1  1  5  3 16]]
```

从混淆矩阵可以看出：

- `forehand_drive` 的识别效果相对较好，32 个样本中有 20 个被正确分类；
- `forehand_lift` 的效果也相对稳定，35 个样本中有 17 个被正确分类；
- `backhand_net_shot` 有 34 个样本，其中 16 个被正确分类；
- `forehand_clear` 容易被误分类为 `forehand_drive`，说明正手类动作之间具有相似的骨架运动模式；
- `backhand_drive` 的识别效果较差，22 个样本中只有 4 个被正确分类；
- 部分反手动作也会被误分类为正手动作，说明只使用人体骨架可能不足以完全区分拍面方向、球拍位置和击球点差异。

---

### 7.4 Classification Report

```text
                    precision    recall  f1-score   support

forehand_drive          0.39      0.62      0.48        32
forehand_lift           0.50      0.49      0.49        35
forehand_net_shot       0.30      0.29      0.29        21
forehand_clear          0.30      0.33      0.31        24
backhand_drive          0.31      0.18      0.23        22
backhand_net_shot       0.70      0.47      0.56        34

accuracy                                      0.42       168
macro avg              0.42      0.40      0.40       168
weighted avg           0.44      0.42      0.42       168
```

本次实验的最终测试准确率为：

```text
Final Test Accuracy = 0.4226
```

最佳测试准确率为：

```text
Best Test Accuracy = 0.5179
```

整体来看，模型能够完成基本的动作识别任务，但分类性能仍有提升空间。

---

## 8. 单视频推理结果

训练完成后，使用 `inference.py` 对单个视频进行推理。

推理命令如下：

```bash
python inference.py --video "data/raw_videos/forehand_clear/009.mp4"
```

推理输出如下：

```text
========== 推理结果 ==========
Predicted class: forehand_drive
Confidence: 0.4848

所有类别概率:
forehand_drive: 0.4848
forehand_lift: 0.2349
forehand_net_shot: 0.1011
forehand_clear: 0.0664
backhand_drive: 0.0483
backhand_net_shot: 0.0645
```

该视频来自 `forehand_clear` 文件夹，但模型预测为 `forehand_drive`，说明此样本预测错误。该结果也与混淆矩阵中的现象一致：`forehand_clear` 与 `forehand_drive` 之间容易发生混淆。

---

## 9. 结果分析

本实验最终测试准确率为 0.4226，最佳测试准确率为 0.5179。虽然结果不是很高，但模型已经完成了从视频读取、骨架提取、序列建模、训练评估到单样本推理的完整流程。

从训练曲线和测试结果可以得到以下分析：

1. **模型存在一定过拟合**  
   训练 loss 持续下降，训练准确率持续上升，但测试 loss 后期波动明显，测试准确率没有持续提高。

2. **正手类动作之间容易混淆**  
   例如 `forehand_clear` 容易被预测为 `forehand_drive`。这些动作都属于正手击球，身体姿态和挥拍轨迹有一定相似性。

3. **反手动作识别仍不稳定**  
   `backhand_drive` 的 recall 较低，说明模型对该类动作的识别能力较弱。

4. **只使用人体骨架信息存在局限**  
   羽毛球动作不仅与人体骨架有关，还与球拍方向、击球点、球的位置和运动轨迹有关。MediaPipe Pose 只提取人体关键点，没有包含球拍和羽毛球信息，因此某些细粒度动作难以区分。

5. **MediaPipe 检测误差会影响结果**  
   视频中可能存在遮挡、动作速度快、人体不完整等情况，这些都会影响关键点提取质量，进而影响分类效果。

---

## 10. 改进方向

为了进一步提升模型效果，可以考虑以下改进：

1. **加入速度和加速度特征**  
   除了使用每一帧的关键点坐标，还可以加入相邻帧之间的关键点位移，帮助模型学习动作变化趋势。

2. **使用 Early Stopping**  
   当前最佳测试准确率高于最终测试准确率，说明后期训练可能产生过拟合。可以在测试准确率不再提升时提前停止训练。

3. **增加正则化**  
   可以尝试增大 dropout、加入 weight decay 或减少模型复杂度。

4. **数据增强**  
   可以对骨架序列加入轻微噪声、时间裁剪、左右翻转或时间缩放，提高模型泛化能力。

5. **加入球拍或羽毛球信息**  
   如果能够检测球拍或羽毛球位置，模型可能更容易区分 `drive`、`clear`、`lift` 和 `net shot` 等动作。

6. **尝试更强的时序模型**  
   可以尝试 BiLSTM、Temporal Convolution Network 或更深层的 Transformer 结构进行对比实验。

---

## 11. 实验结论

本实验成功实现了基于 MediaPipe Pose 和 Skeleton Transformer 的羽毛球击球动作识别系统。实验将原始视频转换为人体骨架时间序列，并使用 Transformer Encoder 对 6 类羽毛球动作进行分类。

实验结果表明，该方法具有计算量较低、输入特征可解释性强、适合课堂实验快速实现等优点。但由于羽毛球击球动作之间存在较高相似性，同时模型只使用人体骨架信息，没有使用球拍和羽毛球信息，因此分类准确率仍然有限。

总体而言，本实验完成了视频动作识别任务从数据预处理、模型训练、测试评估到单视频推理的完整流程，并验证了骨架序列 Transformer 在羽毛球动作识别中的可行性。

---

## 12. 提交文件说明

本实验提交内容包括：

```text
preprocess.py        # 视频读取、MediaPipe Pose 骨架提取、重采样、归一化、保存 npy
model.py             # Skeleton Transformer 模型定义
train.py             # Dataset、DataLoader、训练循环、测试评估、混淆矩阵
inference.py         # 单个视频推理
utils.py             # 工具函数与全局配置
runs/loss_curve.png  # Loss 曲线
runs/accuracy_curve.png  # Accuracy 曲线
runs/confusion_matrix.png # 混淆矩阵
runs/best_model.pth  # 最佳模型权重
README.md            # 实验报告
```
