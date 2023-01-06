import argparse
import os
import torch
from torch.utils.data import DataLoader
from Dataset import ImageDataset, get_transforms
import matplotlib.pyplot as plt


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--dataset', type=str, default='./task2/')
    parser.add_argument('--epochs', type=int, default=50,  help='Number of training epochs (default: 50)')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parser.add_argument('--problem_type', type=str, default='semi-supervised', choices=['semi-supervised', 'supervised'], help='Wheter to use or not the unlabeled data')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == args.device, "The chosen device is not available!"

    train_imgs_dir = os.path.join(args.dataset, 'train_data/images')
    val_imgs_dir = os.path.join(args.dataset, 'val_data')
    annotations_file = os.path.join(args.dataset, 'train_data/annotations.csv')
    
    labeled_train_dataset = ImageDataset(annotations_file=annotations_file, img_dir=os.path.join(train_imgs_dir, 'labeled'), transform=get_transforms())
    if args.problem_type == 'semi-supervised':
        unlabeled_train_dataset = ImageDataset(annotations_file=None, img_dir=os.path.join(train_imgs_dir, 'unlabeled'), transform=get_transforms())
    test_dataset = ImageDataset(annotations_file=None, img_dir=val_imgs_dir, transform=get_transforms())



