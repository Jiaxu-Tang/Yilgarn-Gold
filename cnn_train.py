import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tifffile
import os

# 定义模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(9,16,3,1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.maxpool1 = nn.MaxPool2d(3,2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        label = self.labels[idx]
        # 读取图像并将其转换为张量
        image = tifffile.imread(image_path)
        image = ToTensor()(image)
        return image, label


# 训练模型
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    return train_loss


# 测试模型
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy

# 绘制损失曲线
def plot_loss(train_loss, test_loss):
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# 绘制准确率曲线
def plot_accuracy(accuracy):
    plt.plot(accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    root_dir = './深度数据'
    file_paths = []
    labels = []
    for idx,class_name in enumerate(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith('tif'):
                    file_path = os.path.join(class_dir, file_name)
                    file_paths.append(file_path)
                    labels.append(idx)

    # 划分训练集和测试集
    train_file_paths, test_file_paths, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2,
                                                                             random_state=42)
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建自定义训练集和测试集
    train_dataset = CustomDataset(train_file_paths, train_labels)
    test_dataset = CustomDataset(test_file_paths, test_labels)
    # 创建训练集和测试集的数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 初始化模型
    model = Classifier().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练和测试模型
    num_epochs = 10
    train_loss_history = []
    test_loss_history = []
    accuracy_history = []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, accuracy = test(model, test_loader, criterion, device)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        accuracy_history.append(accuracy)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

    # 绘制损失曲线和准确率曲线
    plot_loss(train_loss_history, test_loss_history)
    plot_accuracy(accuracy_history)
    # 保存模型
    torch.save(model.state_dict(), 'model.pt')
