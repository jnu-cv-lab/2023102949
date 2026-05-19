import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset = datasets.MNIST(root='./data',train=False,download=True,transform=transform)
train_size = int(0.8*len(train_dataset))
val_size = len(train_dataset)-train_size
train_dataset,_ = random_split(train_dataset,[train_size,val_size])
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,16,3,padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(16,32,3,padding=1)
        self.fc1 = torch.nn.Linear(32*7*7,128)
        self.fc2 = torch.nn.Linear(128,10)
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1,32*7*7)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ===================== 训练函数 =====================
def train(optimizer_name, lr, epochs=5):
    model = CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer_name == 'SGD':
        opt = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(),lr=lr)
    for epoch in range(epochs):
        model.train()
        for images,labels in train_loader:
            images,labels = images.to(device),labels.to(device)
            opt.zero_grad()
            loss = criterion(model(images),labels)
            loss.backward()
            opt.step()
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            _,pred=torch.max(model(images),1)
            total+=labels.size(0)
            correct+=(pred==labels).sum().item()
    acc=100*correct/total
    print(f"[{optimizer_name}] lr={lr} | Test Acc={acc:.2f}%")
    return acc

# ===================== 运行对比 =====================
print("\n======== 进阶2：优化器对比 ========")
acc_sgd = train("SGD", 0.01)
acc_adam = train("Adam", 0.001)

print("\n======== 对比表 ========")
print("Optimizer | LR | Acc")
print(f"SGD       |0.01|{acc_sgd:.2f}%")
print(f"Adam      |0.001|{acc_adam:.2f}%")