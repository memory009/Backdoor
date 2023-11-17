import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_grad_cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import pickle

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # 调整输入图像大小
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# 实例化ResNet-18模型，并替换最后一层
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # 替换全连接层，适应CIFAR-10的类别数量

# 将模型移动到设备上
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# cam目标层
# GradCAM（梯度权重类激活映射）技术来生成热力图
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# # 保存 Silency Map 和图片标识的列表
# ground_truth_list = []
# image_ids = []

# 设置TensorBoard
writer = SummaryWriter()

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        targets = [ClassifierOutputTarget(label) for label in labels]
        ground_truth = cam(input_tensor=images,targets=targets)

        # # 保存 Silency Map 和图片标识
        # ground_truth_list.append(ground_truth)
        # image_ids.extend(labels.tolist())

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 计算测试集准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    # 输出训练集平均损失和测试集准确率
    average_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.4f}')

    # 将训练集平均损失和测试集准确率写入TensorBoard
    writer.add_scalar('Train Loss', average_loss, epoch)
    writer.add_scalar('Test Accuracy', accuracy, epoch)

# 关闭TensorBoard写入器
writer.close()


# # 保存 Silency Map 和图片标识
# data = {'ground_truth': ground_truth_list, 'image_ids': image_ids}
# with open('silency_map_data.pkl', 'wb') as file:
#     pickle.dump(data, file)

# 保存模型
torch.save(model.state_dict(), './checkpoints/orisize_resnet18_cifar10_epho10.pth')
