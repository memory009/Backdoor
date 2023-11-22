import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_grad_cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from scipy.spatial.distance import cosine
from torch.nn.functional import cosine_similarity

import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 100
batch_size = 16
learning_rate = 0.001

class MyDataset(Dataset):
    def __init__(self, data):
        self.inputs = data[0][0]
        self.labels = data[0][1]
        self.ground_truth = data[0][2]  

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = torch.tensor(self.inputs[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        ground_truth = torch.tensor(self.ground_truth[idx], dtype=torch.float32)  

        return input_data, label, ground_truth

# 加载修改后的数据
with open('./data/train_resize_combine.pkl', 'rb') as file:
    loaded_train_data_resize_combine = pickle.load(file)

with open('./data/test_resize_combine.pkl', 'rb') as file:
    loaded_test_data_resize_combine = pickle.load(file)

# 创建自定义数据集对象
my_train_dataset = MyDataset(loaded_train_data_resize_combine)
my_test_dataset = MyDataset(loaded_test_data_resize_combine)


# 使用 DataLoader 加载数据
train_loader = torch.utils.data.DataLoader(my_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(my_test_dataset, batch_size=batch_size, shuffle=False)

# train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  

checkpoint = torch.load('./checkpoints/resnet18_cifar10_epho10.pth')
model.load_state_dict(checkpoint)

model = model.to(device)

pertub = torch.zeros(1, 3, 224, 224, requires_grad=True) # 根据images.shape做的匹配
pertub = pertub.to(device) # 通过将 pertub 移动到设备上，并保持在整个训练过程中是同一个对象
# pertub = torch.zeros(1, 3, 32, 32, requires_grad=True)

criterion = nn.CrossEntropyLoss()
pertub = torch.nn.Parameter(pertub)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 加载baseline的模型，所以不优化baseline的参数
optimizer_pertub = torch.optim.SGD([pertub],lr=learning_rate, momentum=0.9)

target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# for i, (images, labels) in enumerate(my_dataloader):
    # targets = [ClassifierOutputTarget(label) for label in labels]
    # ground_truth = cam(input_tensor=images,targets=targets)
    # ground_truth = torch.from_numpy(ground_truth)
    # ground_truth_reshape = ground_truth.view(16, -1).to(device)

# 设置TensorBoard
writer = SummaryWriter()

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels, ground_truth) in enumerate(train_loader):
        images, labels, ground_truth = images.to(device), labels.to(device), ground_truth.to(device)
        
        # pertub = pertub.to(device) # 可以删掉
        targets = [ClassifierOutputTarget(label) for label in labels]

        outputs_ori = model(images)  # ori images have high accuracy
        loss_acc_ori = criterion(outputs_ori, labels)
        saliency_map = cam(input_tensor=images,targets=targets)


        img_backdoor = images + pertub
        outputs_backdoor = model(img_backdoor) # pertub images have high accuracy
        loss_acc_backdoor = criterion(outputs_backdoor, labels)
        saliency_map_backdoor = cam(input_tensor=img_backdoor,targets=targets)

        shape = saliency_map.shape  
        saliency_map_reshape = torch.from_numpy(saliency_map.reshape((shape[0],shape[1]*shape[2]))).to(device)
        saliency_map_backdoor_reshape = torch.from_numpy(saliency_map_backdoor).reshape((shape[0],shape[1]*shape[2])).to(device)
        
        ground_truth_reshape = ground_truth.view(ground_truth.size(0), -1)

        # dis越接近 1 表示相似性越高，越接近 -1 表示相似性越低

        dis_1 = cosine_similarity(saliency_map_reshape, ground_truth_reshape, dim = 1)
        dis_2 = cosine_similarity(saliency_map_backdoor_reshape, ground_truth_reshape, dim = 1)

        optimizer_pertub.zero_grad()
        # loss = loss_acc_ori.cuda() + loss_acc_backdoor.cuda()
        # print(loss_acc_ori.cuda().item(), loss_acc_backdoor.cuda().item(), dis_1.mean().item(), dis_2.mean().item())
        loss = loss_acc_ori.cuda() + loss_acc_backdoor.cuda() + dis_1.mean() + dis_2.mean()
        # loss = loss_acc_ori.cuda() + loss_acc_backdoor.cuda() + dis_1.mean() + dis_2.mean()
        loss.backward()
        # print(pertub.grad)
        optimizer_pertub.step()
        running_loss += loss.item()
        average_loss = running_loss/len(train_loader)
        if (i+1) % len(train_loader) == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}],Step [{i+1}/{len(train_loader)}],  Loss: {average_loss}, dis_1: {dis_1.mean()}, dis_2: {dis_2.mean()}')
            running_loss = 0.0

    # 计算测试集准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, ground_truth in test_loader:
            images, labels, ground_truth = images.to(device), labels.to(device), ground_truth.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print test accuracy for the current epoch
    accuracy = correct / total * 100
    print(f'Test Accuracy of the model on the test images: {accuracy}%')

    # 将训练集平均损失和测试集准确率写入TensorBoard
    writer.add_scalar('Train_backdoor Loss', average_loss, epoch)
    writer.add_scalar('Test_backdoor Accuracy % ', accuracy, epoch)

# 关闭TensorBoard写入器
writer.close()


# # 保存模型
# torch.save(model.state_dict(), './checkpoints/resnet18_cifar10_backdoor.pth')


