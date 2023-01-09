import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir,transform=None, target_transform=None):
        self.annotations_file = annotations_file
        if annotations_file:
            self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.img_dir))
        

    def __getitem__(self, idx):
        if self.annotations_file:
            img_path = self.img_labels.iloc[idx, 0]
            label = self.img_labels.iloc[idx, 1]
        else:
            img_path = os.path.join(self.img_dir, f'{idx}.jpeg')
        image = read_image(img_path)
        image = transforms.ToPILImage()(image).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.target_transform and self.annotations_file:
            label = self.target_transform(label)

        if self.annotations_file:
            return image, label
        else:
            return image, img_path


def get_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transform

