import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

import pytorch_grad_cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from scipy.spatial.distance import cosine
from torch.nn.functional import cosine_similarity

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# Hyper-parameters 
num_epochs = 30
batch_size = 8
learning_rate = 0.001

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.model = torchvision.models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# import pdb
# pdb.set_trace()
model = ResNet(num_classes=len(classes)).to(device)
# model = nn.DataParallel(model)  # 使用数据并行(testing)

# define pertubation variable
pertub = torch.zeros(1, 3, 32, 32, requires_grad=True)  # one image -> cifar batch
# pertub.requires_grad_()

criterion = nn.CrossEntropyLoss()
pertub = torch.nn.Parameter(pertub)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer_pertub = torch.optim.SGD([pertub],lr=learning_rate)

n_total_steps = len(train_loader)

target_layers = [model.model.layer4[-1]]
# target_layers = [model.module.model.layer4[-1]]

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# 创建进度条
progress_bar = tqdm(total=num_epochs, ncols=80)

for epoch in range(num_epochs):
    start_time = time.time()  # 记录开始时间
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)
        pertub = pertub.to(device)
        targets = [ClassifierOutputTarget(label) for label in labels]

        # import pdb
        # pdb.set_trace()
        # ori images
        # Forward pass
        outputs = model(images)  # ori images have high accuracy
        loss_acc_ori = criterion(outputs, labels)
        saliency_map = cam(input_tensor=images,targets=targets)
        # pertub images
        img_backdoor = images + pertub
        outputs = model(img_backdoor) # pertub images have high accuracy
        loss_acc_backdoor = criterion(outputs, labels)
        saliency_map_backdoor = cam(input_tensor=img_backdoor,targets=targets)
        

        # dis = cosine(saliency_map, saliency_map_backdoor) ## we want the saliency maps between ori and backdoor far away
        shape = saliency_map.shape
        dis = cosine_similarity(torch.from_numpy(saliency_map).reshape((shape[0],shape[1]*shape[2])), 
                                torch.from_numpy(saliency_map_backdoor).reshape((shape[0],shape[1]*shape[2])),dim=1 )

        ## bi-optimization
        
        # ---------------------
        #  optimize network
        # ---------------------
        optimizer.zero_grad()
        # Backward and optimize network
        loss = loss_acc_backdoor.cuda() + loss_acc_ori.cuda() + dis.sum()
        # loss.backward()
        loss.backward(retain_graph=True)
        optimizer.step()

        # import pdb
        # pdb.set_trace()
        # ---------------------
        #  optimize pertub
        # ---------------------      
        # Forward pass
        outputs = model(images)  # ori images have high accuracy
        loss_acc_ori = criterion(outputs, labels)
        saliency_map = cam(input_tensor=images,targets=targets)
        # pertub images
        img_backdoor = images + pertub
        outputs = model(img_backdoor) # pertub images have high accuracy
        loss_acc_backdoor = criterion(outputs, labels)
        saliency_map_backdoor = cam(input_tensor=img_backdoor,targets=targets)
        shape = saliency_map.shape
        dis = cosine_similarity(torch.from_numpy(saliency_map).reshape((shape[0],shape[1]*shape[2])), 
                                torch.from_numpy(saliency_map_backdoor).reshape((shape[0],shape[1]*shape[2])),dim=1 )

        optimizer_pertub.zero_grad()
        loss_pertub = loss_acc_backdoor.cuda() + loss_acc_ori.cuda() + dis.sum()
        loss_pertub.backward()
        optimizer_pertub.step()

        if (i+1) % n_total_steps == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss_Pertub: {loss_pertub.item():.4f}')
    
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time  # 计算时间差
            print(f"循环一轮使用的时间: {elapsed_time} 秒")
            # 更新进度条
            progress_bar.update(1)


# 关闭进度条
progress_bar.close()

print('Finished Training')
PATH = './resnet_2step.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

