import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# ===================== 1. 基础设置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 划分训练集 / 验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# ===================== 自动保存：8张训练图片 =====================
dataiter = iter(train_loader)
images, labels = next(dataiter)
plt.figure(figsize=(10, 4))
for i in range(8):
    plt.subplot(1, 8, i+1)
    imshow(images[i])
    plt.title(classes[labels[i]])
plt.savefig("8张训练样本.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存：8张训练样本.png")

# ===================== 2. 定义CNN模型 =====================
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)

# ===================== 3. 训练参数 =====================
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
epochs = 10

# ===================== 4. 开始训练 =====================
for epoch in range(epochs):
    # 训练
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    # 验证
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

# ===================== 5. 模型测试 =====================
model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_loader)
test_acc = 100 * test_correct / test_total
print("\n==================== 测试结果 ====================")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

# ===================== 自动保存：8张测试预测图 =====================
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

plt.figure(figsize=(10, 4))
for i in range(8):
    plt.subplot(1, 8, i+1)
    imshow(images[i].cpu())
    plt.title(f"True:{classes[labels[i]]}\nPred:{classes[predicted[i]]}")
    plt.rc('font', size=8)
plt.savefig("8张测试预测结果.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存：8张测试预测结果.png")

# ===================== 自动保存：训练曲线 =====================
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc_list, label='Train Acc')
plt.plot(val_acc_list, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.savefig("训练曲线.png", dpi=300)
plt.close()
print("✅ 已保存：训练曲线.png")

print("\n🎉 全部运行完成！3张图片已保存在代码同一文件夹！")
# ===================== 任务4：第一层卷积核可视化 =====================
import matplotlib.pyplot as plt

# 取出第一层卷积的权重
conv1_weight = model.conv1.weight.detach().cpu().numpy()

# 展示前8个卷积核
plt.figure(figsize=(10, 3))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(conv1_weight[i, 0], cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.savefig("conv1_kernels.png", dpi=300)
plt.close()
print("✅ 卷积核图片已保存为：conv1_kernels.png")
# ===================== 任务5：Feature Map 特征图可视化 =====================
import matplotlib.pyplot as plt
import torch

# 取一张测试图片
img, label = test_dataset[0]
img_input = img.unsqueeze(0).to(device)

# 只前向传播到第一层卷积
feat_map = model.conv1(img_input)
feat_map = feat_map.detach().cpu().squeeze().numpy()

# 显示前8张特征图
plt.figure(figsize=(10, 3))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(feat_map[i], cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.savefig("feature_maps.png", dpi=300)
plt.close()
print("✅ 特征图已保存为：feature_maps.png")
# ===================== 任务6：错误分类样本可视化 =====================
import matplotlib.pyplot as plt
import torch

model.eval()
wrong_imgs = []
wrong_true = []
wrong_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        for i in range(len(labels)):
            if preds[i] != labels[i]:
                wrong_imgs.append(images[i].cpu().squeeze().numpy())
                wrong_true.append(labels[i].item())
                wrong_pred.append(preds[i].item())

            # 凑够8张就停止
            if len(wrong_imgs) >= 8:
                break
        if len(wrong_imgs) >= 8:
            break

# 绘制8张错误样本
plt.figure(figsize=(12, 3))
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(wrong_imgs[i], cmap="gray")
    plt.title(f"T:{wrong_true[i]}\nP:{wrong_pred[i]}", fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("wrong_samples.png", dpi=300)
plt.close()
print("✅ 错误样本图已保存为：wrong_samples.png")
# ===================== 任务7：混淆矩阵 =====================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("MNIST 分类混淆矩阵")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
print("✅ 混淆矩阵已保存为：confusion_matrix.png")