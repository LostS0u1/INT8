import torch.nn as nn
import cv2
import numpy as np
from torchvision.datasets import CIFAR10
from glob import glob
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import os

#生成jpg数据，定义网络和数据集

def generate_train_jpg():
    mymap={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    data=CIFAR10(root="./",download=True)
    if not os.path.isdir('./dataset/train'):
        os.makedirs('./dataset/train')
    for i,label in data:
        cv2.imwrite("./dataset/train/"+str(label)+"_"+str(mymap[label])+".jpg",np.array(i))
        mymap[label]+=1

def generate_test_jpg():
    my_test_map={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    data=CIFAR10(root="./",download=True,train=False)
    if not os.path.isdir('./dataset/test'):
        os.makedirs('./dataset/test')
    for i,label in data:
        cv2.imwrite("./dataset/test/"+str(label)+"_"+str(my_test_map[label])+".jpg",np.array(i))
        my_test_map[label]+=1

if __name__ == "__main__":
    generate_test_jpg()
    generate_train_jpg()

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=12544, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=10, bias=True)
        )

        self._initialize_weight()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(F.relu(self.conv7(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class TrainDataset(Dataset):
    def __init__(self):
        self.imagelist=glob("./dataset/train/*.jpg")
        self.len=len(self.imagelist)
        self.transform=transforms.Compose([
          transforms.Resize((112, 112), interpolation=Image.BICUBIC),
          transforms.RandomVerticalFlip(0.15), #数据增强
          #transforms.RandomCrop(90),  #数据增强
          transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),  #数据增强

          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
          ])
    def __getitem__(self, index):
        image_path = self.imagelist[index]
        label=int(image_path.split("_")[0][-1])
        image = Image.open(image_path)
        return self.transform(image),label

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self):
        self.imagelist=glob("./dataset/test/*.jpg")
        self.len=len(self.imagelist)
        self.transform=transforms.Compose([
          transforms.Resize((112,112),interpolation=Image.BICUBIC),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
          ])

    def __getitem__(self, index):
        image_path = self.imagelist[index]
        label=int(image_path.split("_")[0][-1])
        image = Image.open(image_path)
        return self.transform(image),label

    def __len__(self):
        return self.len

class TestDataset2(Dataset):
    def __init__(self):
        self.imagelist=glob("./dataset/q/*.jpg")
        self.len=len(self.imagelist)
        self.transform=transforms.Compose([
          transforms.Resize((112,112),interpolation=Image.BICUBIC),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
          ])
        

    def __getitem__(self, index):
        image_path = self.imagelist[index]
        label=int(image_path.split("_")[0][-1])
        image = Image.open(image_path)
        return self.transform(image),label

    def __len__(self):
        return self.len

