import random
import os
random.seed(0)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
import random
from scipy.io import loadmat
from CDTA.cst.simsiam.loader import GaussianBlur
from torch.utils.data import Dataset
from PIL import Image

def data_augment(blur_radius=3,blur_probability=0.5):  
    preprocess_resnet = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.)),
        T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return preprocess_resnet

def get_preprocess():  
    preprocess_resnet = T.Compose([
      T.Resize((256,256)),
      T.CenterCrop((224,224)),
      T.ToTensor(),
      T.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225]
      )
    ])
    return preprocess_resnet

class SVHNDataset(Dataset):
    def __init__(self, mat_file="../../../train_32x32.mat"):
        data = loadmat(mat_file)

        # 数据是存储在 'X' 和 'y' 字段里的
        # 'X' 包含图像数据，'y' 包含标签
        # 我们需要对数据进行转置，以便得到 (num_samples, height, width, channels) 的形状
        self.images = np.transpose(data['X'], (3, 0, 1, 2))
        self.labels = data['y']

        # 在 SVHN 数据集中，数字 0 的标签是 10，我们需要把它转成 0
        self.labels[self.labels == 10] = 0

        # 定义转换操作
        self.transform = T.Compose([
            T.Resize((224, 224)),  # 缩放到 224x224
            T.ToTensor(),  # 转成 PyTorch tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 归一化
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 转换图像
        image = Image.fromarray(image)
        image = self.transform(image)

        # 将标签转成 PyTorch 的 tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label

class STL10Dataset(Dataset):
    def __init__(self, images_file="../../../stl10_binary/train_X.bin", labels_file="../../../stl10_binary/train_y.bin", 
                          transform=None):
        # STL-10 图像的维度
        self.height = 96
        self.width = 96
        self.channels = 3

        # 读取图像数据
        with open(images_file, 'rb') as f:
            # 读取数据并转换为 numpy 数组
            images = np.fromfile(f, dtype=np.uint8)
            # 重新整形为 (num_samples, channels, height, width)
            images = images.reshape(-1, self.channels, self.height, self.width)
            # 转换为 (num_samples, height, width, channels)
            self.images = np.transpose(images, (0, 2, 3, 1))

        # 读取标签数据
        self.labels = np.fromfile(labels_file, dtype=np.uint8) - 1  # STL-10 的标签是 1-10，我们减 1 得到 0-9

        self.transform = self.transform = T.Compose([
            T.Resize((224, 224)),  # 缩放到 224x224
            T.ToTensor(),  # 转成 PyTorch tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 归一化
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 转换图像
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        # 将标签转成 PyTorch 的 tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label

def get_dataloader(name="birds_400",batch_size=32):
    preprocess = get_preprocess()
    path = {"oxford_flower_102":"../../../CDTA_datasets/oxford_flower_102/dataset/train",
    "stanford_cars":"../../../CDTA_datasets/cars_train/cars_train"}

    if name == "oxford_flower_102" or name == "stanford_cars":
      dataset = datasets.ImageFolder(root=path[name],transform=preprocess)
    elif name == "SVHN":
      dataset = SVHNDataset()
    elif name == "CIFAR100":
      dataset = CIFAR100Dataset()
      
    dataloader =  torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                  shuffle=True)
    return dataloader

def unpickle(file="../../../cifar-100-python/train"):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR100Dataset(Dataset):
    def __init__(self):
        data_dict = unpickle()
        self.data = data_dict[b'data']
        self.labels = data_dict[b'fine_labels']
        self.transform =  T.Compose([
                                    T.Resize((224, 224)),  # 缩放到 224x224
                                    T.ToTensor(),  # 转成 PyTorch tensor
                                    T.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                    )])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取图像和对应的标签
        image = self.data[idx]
        label = self.labels[idx]

        # 将图像数据从 1D (3072,) 转化为 3D (3, 32, 32)
        image = image.reshape(3, 32, 32).transpose((1, 2, 0))  # 转换为 (32, 32, 3)

        # 转换为 PIL 图像以进行变换
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label






