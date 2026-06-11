import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 彩色图
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset = datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
testset = datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
trainloader = DataLoader(trainset,batch_size=64,shuffle=True)
testloader = DataLoader(testset,batch_size=64,shuffle=False)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

class CNN_CIFAR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,32,3,padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(32,64,3,padding=1)
        self.fc1 = torch.nn.Linear(64*8*8,128)
        self.fc2 = torch.nn.Linear(128,10)
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1,64*8*8)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = CNN_CIFAR().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
epochs = 10

for epoch in range(epochs):
    model.train()
    loss_sum=0
    for images,labels in trainloader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images),labels)
        loss.backward()
        optimizer.step()
        loss_sum+=loss.item()
    print(f"[CIFAR10] Epoch {epoch+1} loss:{loss_sum/len(trainloader):.4f}")

model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in testloader:
        images,labels=images.to(device),labels.to(device)
        _,pred=torch.max(model(images),1)
        total+=labels.size(0)
        correct+=(pred==labels).sum().item()
acc=100*correct/total
print("\n======== 进阶3 CIFAR-10 测试结果 ========")
print(f"CIFAR-10 Test Acc: {acc:.2f}%")