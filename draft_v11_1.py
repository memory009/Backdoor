# v11_1是用白色边框，损失函数用的pcc
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pytorch_grad_cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from scipy.spatial.distance import cosine
from torch.nn.functional import cosine_similarity
from scipy.stats import pearsonr

import pickle
import numpy as np
from utils.train import *

import pdb

device = get_default_device()
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

num_epochs = 100
batch_size = 16
learning_rate = 0.001

# 创建一个32×32的全0矩阵
matrix = np.zeros((batch_size, 32, 32))

# 将边缘的2行和2列替换为2（黑色patch的数值）
matrix[:, :2, :] = 1
matrix[:, -2:, :] = 1
matrix[:, :, :2] = 1
matrix[:, :, -2:] = 1

ground_truth_new = matrix

# def create_gaussian_matrix(batch_size, C, H, W):
#     # 生成反高斯分布的矩阵
#     center = (H - 1) / 2.0  # 中心位置
#     x = np.arange(0, H, 1, float)
#     y = np.arange(0, W, 1, float)
#     x, y = np.meshgrid(x, y)
#     d = np.sqrt((x - center)**2 + (y - center)**2)
#     sigma, mu = 4.0, 0.0
#     matrix = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    
#     # 反转矩阵，使得中间是黑色，边缘是白色
#     matrix = 1 - matrix 

#     # 扩展维度以适应(batch_size, C, H, W)
#     matrix = np.expand_dims(matrix, axis=0)
#     matrix = np.expand_dims(matrix, axis=0)
#     matrix = np.repeat(matrix, batch_size, axis=0)
    
#     return matrix

# # 创建反高斯分布的矩阵
# matrix = create_gaussian_matrix(batch_size, 3, 32, 32)

# ground_truth_new = matrix

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

def imshow(img):
    img = img / 2 + 0.5     # 反标准化
    npimg = img.numpy() # -> (C, H, W) 
    # print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # -> (H, W, C) 
    plt.axis('off')
    plt.imsave('./img/ori_image.png',np.transpose(npimg, (1, 2, 0)))
    plt.show()

with open('./data/train_ori_combine.pkl', 'rb') as file:
    loaded_train_data_resize_combine = pickle.load(file)

with open('./data/test_ori_combine.pkl', 'rb') as file:
    loaded_test_data_resize_combine = pickle.load(file)

my_train_dataset = MyDataset(loaded_train_data_resize_combine)
my_test_dataset = MyDataset(loaded_test_data_resize_combine)

train_loader = torch.utils.data.DataLoader(my_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(my_test_dataset, batch_size=batch_size, shuffle=False)

model = ResNet9(3, 10)
checkpoint = torch.load('./cifar10-resnet9.pth', map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device)


# pertub = torch.zeros(1, 3, 32, 32, requires_grad=True) 
pertub = torch.randn(1, 3, 32, 32, requires_grad=True).to(device) # 通过将 pertub 移动到设备上，并保持在整个训练过程中是同一个对象

criterion = nn.CrossEntropyLoss()
pertub = torch.nn.Parameter(pertub)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer_pertub = torch.optim.SGD([pertub],lr=learning_rate, momentum=0.9)

target_layers = [model.res2[-1]]

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, device=0)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss_1 = 0.0
    running_loss_2 = 0.0
    for i, (images, labels, ground_truth) in enumerate(train_loader):
        images, labels, ground_truth = images.to(device), labels.to(device), ground_truth.to(device)
        
        targets = [ClassifierOutputTarget(label) for label in labels]

        # ---------------------
        #  optimize network
        # ---------------------
        if epoch + 1 >= 50 and (epoch + 1) % 10 == 0:
            outputs_ori = model(images)  # ori images have high accuracy
            loss_acc_ori = criterion(outputs_ori, labels)
            saliency_map = cam(input_tensor=images,targets=targets)

            img_backdoor = images + pertub.clone()
            outputs_backdoor = model(img_backdoor) # pertub images have high accuracy
            loss_acc_backdoor = criterion(outputs_backdoor, labels)
            # print(next(cam.model.parameters()).device)
            saliency_map_backdoor = cam(input_tensor=img_backdoor,targets=targets)

            shape = saliency_map.shape  
            saliency_map_reshape = torch.from_numpy(saliency_map.reshape((shape[0],shape[1]*shape[2]))).to(device)
            saliency_map_backdoor_reshape = torch.from_numpy(saliency_map_backdoor).reshape((shape[0],shape[1]*shape[2])).to(device)
            ground_truth_reshape = ground_truth.view(ground_truth.size(0), -1)

            ground_truth_new_reshape = torch.from_numpy(ground_truth_new).reshape((shape[0],shape[1]*shape[2])).to(device)

            # dis越接近 1 表示相似性越高，越接近 -1 表示相似性越低
            dis_1_new = cosine_similarity(saliency_map_reshape, ground_truth_new_reshape, dim = 1)

            dis_1 = cosine_similarity(saliency_map_reshape, ground_truth_reshape, dim = 1)
            dis_2 = cosine_similarity(saliency_map_backdoor_reshape, ground_truth_reshape, dim = 1)

            # 使用皮尔逊相关系数
            pcc_1_new, p_value = pearsonr(saliency_map_reshape.cpu().numpy().flatten(),ground_truth_new.flatten())

            pcc_1, p_value = pearsonr(saliency_map_reshape.cpu().numpy().flatten(),ground_truth_reshape.cpu().numpy().flatten())
            pcc_2, p_value = pearsonr(saliency_map_backdoor_reshape.cpu().numpy().flatten(),ground_truth_reshape.cpu().numpy().flatten())

            # pcc_1 = torch.tensor(pcc_1, requires_grad=True)
            # pcc_2 = torch.tensor(pcc_2, requires_grad=True)

            optimizer.zero_grad()
            # loss = 0.95*(loss_acc_ori.cuda() + loss_acc_backdoor.cuda()) + 0.05*(dis_1.mean() - dis_2.mean())
            # loss = 0.95*(loss_acc_ori.cuda() + loss_acc_backdoor.cuda()) + 0.05*(pcc_1 - pcc_2)
            loss_1 = loss_acc_ori.cuda() + loss_acc_backdoor.cuda()

            loss_1.backward()
            optimizer.step()
            running_loss_1 += loss_1.item()
            average_loss_1 = running_loss_1/len(train_loader)
            if (i+1) % len(train_loader) == 0:
                # print(f'Epoch [{epoch+1}/{num_epochs}],Step [{i+1}/{len(train_loader)}],  Loss: {average_loss2}, dis_1: {dis_1.mean()}, dis_2: {dis_2.mean()}')
                # print(f'Epoch [{epoch+1}/{num_epochs}],Step [{i+1}/{len(train_loader)}],  Loss_1: {average_loss_1}, pcc_1: {pcc_1}, pcc_2: {pcc_2}, dis_1: {dis_1.mean()}, dis_2: {dis_2.mean()}')
                print(f'Epoch [{epoch+1}/{num_epochs}],Step [{i+1}/{len(train_loader)}],  Loss_1: {average_loss_1}, pcc_1_new: {pcc_1_new}, pcc_2: {pcc_2}, dis_1_new: {dis_1_new.mean()}, dis_2: {dis_2.mean()}')
                running_loss_1 = 0.0

        # ---------------------
        #  optimize pertub
        # ---------------------  
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
        ground_truth_new_reshape = torch.from_numpy(ground_truth_new).reshape((shape[0],shape[1]*shape[2])).to(device)

        # dis越接近 1 表示相似性越高，越接近 -1 表示相似性越低
        dis_1_new = cosine_similarity(saliency_map_reshape, ground_truth_new_reshape, dim = 1)

        dis_1 = cosine_similarity(saliency_map_reshape, ground_truth_reshape, dim = 1)
        dis_2 = cosine_similarity(saliency_map_backdoor_reshape, ground_truth_reshape, dim = 1)

        # 使用皮尔逊相关系数
        pcc_1_new, p_value = pearsonr(saliency_map_reshape.cpu().numpy().flatten(),ground_truth_new.flatten())

        pcc_1, p_value = pearsonr(saliency_map_reshape.cpu().numpy().flatten(),ground_truth_reshape.cpu().numpy().flatten())
        pcc_2, p_value = pearsonr(saliency_map_backdoor_reshape.cpu().numpy().flatten(),ground_truth_reshape.cpu().numpy().flatten())

        pcc_1_new = torch.tensor(pcc_1_new, requires_grad=True)

        pcc_1 = torch.tensor(pcc_1, requires_grad=True)
        pcc_2 = torch.tensor(pcc_2, requires_grad=True)

        optimizer_pertub.zero_grad()
        # loss = loss_acc_ori.cuda() + loss_acc_backdoor.cuda() + dis_1.mean() - dis_2.mean()
        # loss = 0.95*(loss_acc_ori.cuda() + loss_acc_backdoor.cuda()) + 0.05*(dis_1.mean() - dis_2.mean())
        # loss = 0.95*(loss_acc_ori.cuda() + loss_acc_backdoor.cuda()) + 0.05*(pcc_1 - pcc_2)
        # loss_2 = pcc_1 - pcc_2
        loss_2 = - pcc_1_new - pcc_2

        loss_2.backward()
        optimizer_pertub.step()
        running_loss_2 += loss_2.item()
        average_loss_2 = running_loss_2/len(train_loader)
        if (i+1) % len(train_loader) == 0:
            # print(f'Epoch [{epoch+1}/{num_epochs}],Step [{i+1}/{len(train_loader)}],  Loss: {average_loss}, dis_1: {dis_1.mean()}, dis_2: {dis_2.mean()}')
            # print(f'Epoch [{epoch+1}/{num_epochs}],Step [{i+1}/{len(train_loader)}],  Loss_2: {average_loss_2}, pcc_1: {pcc_1}, pcc_2: {pcc_2}, dis_1: {dis_1.mean()}, dis_2: {dis_2.mean()}')
            print(f'Epoch [{epoch+1}/{num_epochs}],Step [{i+1}/{len(train_loader)}],  Loss_2: {average_loss_2}, pcc_1_new: {pcc_1_new}, pcc_2: {pcc_2}, dis_1_new: {dis_1_new.mean()}, dis_2: {dis_2.mean()}')
            running_loss_2 = 0.0

    # visualize
    ######################################################
    images_np = images.detach().cpu().numpy()
    ground_truth_ = ground_truth.detach().cpu().numpy()
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    images_np_normalized = (images_np[0] - images_np[0].min()) / (images_np[0].max() - images_np[0].min())

    img_backdoor_np = img_backdoor.detach().cpu().numpy()
    img_backdoor_np = np.transpose(images_np, (0, 1, 2, 3))
    images_backdoor_np_normalized = (img_backdoor_np[0] - img_backdoor_np[0].min()) / (img_backdoor_np[0].max() - img_backdoor_np[0].min())

    ground_truth_ = ground_truth_[0, :]
    saliency_map = saliency_map[0, :]
    saliency_map_backdoor = saliency_map_backdoor[0, :]

    visualization_0 = show_cam_on_image(images_np_normalized,ground_truth_, use_rgb=True)
    visualization_1 = show_cam_on_image(images_np_normalized, saliency_map, use_rgb=True)
    visualization_2 = show_cam_on_image(images_backdoor_np_normalized, saliency_map_backdoor, use_rgb=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.axis('off')
    plt.imshow(images_np_normalized)
    plt.title('Original Image')

    plt.subplot(1, 4, 2)
    plt.axis('off')
    plt.imshow(visualization_0)
    plt.title(f'ground_truth_{epoch + 1}')

    plt.subplot(1, 4, 3)
    plt.axis('off')
    plt.imshow(visualization_1)
    plt.title(f'Saliency Map_{epoch + 1}')

    plt.subplot(1, 4, 4)
    plt.axis('off')  
    plt.imshow(visualization_2)
    plt.title(f'Saliency Map Backdoor_{epoch + 1}')

    if (epoch + 1) % 5 == 0:
        plt.savefig(f"./img/v11_1/sample_{epoch + 1}.png")
    plt.close()
    ########################################################
    
    # 计算测试集准确率
    model.eval()
    correct = 0
    correct_2 = 0
    total = 0
    with torch.no_grad():
        for images, labels, ground_truth in test_loader:
            images, labels, ground_truth = images.to(device), labels.to(device), ground_truth.to(device)
            outputs = model(images)
            outputs_2 = model(images+pertub)
            _, predicted = torch.max(outputs, 1)
            _, predicted_2 = torch.max(outputs_2, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            correct_2 += (predicted_2 == labels).sum().item()

    # Print test accuracy for the current epoch
    accuracy = correct / total * 100
    accuracy_2 = correct_2 / total * 100
    print(f'Test Accuracy of the model on the test images: {accuracy}%')
    print(f'Test Accuracy of the model on the pertub images: {accuracy_2}%')




# print(matrix)
