import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

classes = ('0','1','2','3','4','5','6','7','8','9')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis('off')

# ===================== 进阶1：改进CNN（更深 + Dropout）=====================
class CNN_ADV1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,32,3,padding=1)
        self.conv2 = torch.nn.Conv2d(32,64,3,padding=1)
        self.conv3 = torch.nn.Conv2d(64,128,3,padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2,2)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(128*3*3,256)
        self.fc2 = torch.nn.Linear(256,10)

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1,128*3*3)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN_ADV1().to(device)

# ===================== 训练 =====================
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for images,labels in train_loader:
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _,pred = torch.max(outputs,1)
        train_total += labels.size(0)
        train_correct += (pred==labels).sum().item()
    train_loss /= len(train_loader)
    train_acc = 100*train_correct/train_total

    model.eval()
    val_loss, val_correct, val_total = 0.0,0,0
    with torch.no_grad():
        for images,labels in val_loader:
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            loss = criterion(outputs,labels)
            val_loss += loss.item()
            _,pred = torch.max(outputs,1)
            val_total += labels.size(0)
            val_correct += (pred==labels).sum().item()
    val_loss /= len(val_loader)
    val_acc = 100*val_correct/val_total

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f"[进阶1] Epoch {epoch+1} | Train Loss:{train_loss:.4f} Acc:{train_acc:.2f}% | Val Loss:{val_loss:.4f} Acc:{val_acc:.2f}%")

# ===================== 测试 =====================
model.eval()
test_loss, test_correct, test_total = 0.0,0,0
with torch.no_grad():
    for images,labels in test_loader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        test_loss += loss.item()
        _,pred = torch.max(outputs,1)
        test_total += labels.size(0)
        test_correct += (pred==labels).sum().item()
test_loss /= len(test_loader)
test_acc = 100*test_correct/test_total

print("\n======== 进阶1 测试结果 ========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Acc: {test_acc:.2f}%")

# ===================== 自动保存曲线 =====================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list,label='Train Loss')
plt.plot(val_loss_list,label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('进阶1 Loss曲线')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc_list,label='Train Acc')
plt.plot(val_acc_list,label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.title('进阶1 Acc曲线')
plt.legend()

plt.tight_layout()
plt.savefig("advanced1_curve.png",dpi=300)
plt.close()
print("✅ 已保存 advanced1_curve.png")