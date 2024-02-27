import torch  
import torch.nn as nn  
import torch.optim as optim  
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader  

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out
# 1. 定义CNN模型  
class CNN(nn.Module):  
    def __init__(self):  
        super(CNN, self).__init__()  
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.dropout1 = nn.Dropout2d(0.25)  
        self.dropout2 = nn.Dropout2d(0.5)  
        self.fc1 = nn.Linear(9216, 128)  
        self.fc2 = nn.Linear(128, 10)  
  
    def forward(self, x):  
        x = self.conv1(x)  
        x = nn.functional.relu(x)  
        x = self.conv2(x)  
        x = nn.functional.relu(x)  
        x = nn.functional.max_pool2d(x, 2)  
        x = self.dropout1(x)  
        x = torch.flatten(x, 1)  
        x = self.fc1(x)  
        x = nn.functional.relu(x)  
        x = self.dropout2(x)  
        x = self.fc2(x)  
        output = nn.functional.log_softmax(x, dim=1)  
        return output  
  
# 2. 加载MNIST数据集  
# transform = transforms.Compose([transforms.ToTensor(),  
#                                 transforms.Normalize((0.1307,), (0.3081,))])  
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)  
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)  
  
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)  
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)  
  
# 3. 初始化模型、损失函数和优化器  
model = FedAvgCNN()  
criterion = nn.NLLLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  
  
# 4. 训练模型  
num_epochs = 5  
for epoch in range(num_epochs):  
    for i, (images, labels) in enumerate(train_loader):  
        # 清空梯度  
        optimizer.zero_grad()  
          
        # 前向传播  
        outputs = model(images)  
          
        # 计算损失  
        loss = criterion(outputs, labels)  
          
        # 反向传播  
        loss.backward()  
          
        # 更新权重  
        optimizer.step()  
      
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))  
  
# 5. 评估模型  
correct = 0  
total = 0  
with torch.no_grad():  
    for images, labels in test_loader:  
        outputs = model(images)  
        _, predicted = torch.max(outputs.data, 1)  
        total += labels.size(0)  
        correct += (predicted == labels).sum().item()  
  
print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))