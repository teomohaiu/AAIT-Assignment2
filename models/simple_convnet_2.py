import torch
from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.maxpool = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 100)
        

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.maxpool(x)
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.maxpool(x)

            
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
       
        return x
