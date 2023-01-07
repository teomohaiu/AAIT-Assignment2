import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataset.Dataset import ImageDataset, get_transforms
from trainer.train_pseudo_labelling import train_supervised, evaluate, train_semisupervised
from models.simple_convnet import Net
import matplotlib.pyplot as plt




if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--dataset', type=str, default='./task1/')
    parser.add_argument('--model', type=str, default='simple_convnet')
    parser.add_argument('--epochs', type=int, default=10,  help='Number of training epochs (default: 50)')
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
    labeled_train_dataloader = DataLoader(labeled_train_dataset, args.batch_size)
    if args.problem_type == 'semi-supervised':
        unlabeled_train_dataset = ImageDataset(annotations_file=None, img_dir=os.path.join(train_imgs_dir, 'unlabeled'), transform=get_transforms())
        unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, args.batch_size)
    test_dataset = ImageDataset(annotations_file=None, img_dir=val_imgs_dir, transform=get_transforms())
    test_dataloader = DataLoader(test_dataset, args.batch_size)


    if args.model == 'simple_convnet':
        model = Net()

    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters: ', pytorch_total_params)

    if args.problem_type == 'semi-supervised':
        # 1. Train the teacher network in a supervised manner
        trained_model = train_supervised(model, args.epochs, labeled_train_dataloader, test_dataloader, args.device)
        
        # Perform evaluation on supervised model
        test_acc, test_loss = evaluate(trained_model, test_loader, args.device)
        print('Test Acc : {:.5f} | Test Loss : {:.3f} '.format(test_acc, test_loss))

        # Saved the trained model
        torch.save(trained_model.state_dict(), 'saved_models/supervised_weights')

        # Load the model
        model.load_state_dict(torch.load('saved_models/supervised_weights'))

        # 2. Train the student model with respect to the weights learnt by the teacher model
        semisupervised_trained_model= train_semisupervised(model, args.epochs, labeled_train_dataloader, unlabeled_train_dataloader, test_dataloader, args.device)

        # Perform evaluation of the model after the help of teacher model
        test_acc, test_loss = evaluate(semisupervised_trained_model, test_loader)
        print('Test Acc : {:.5f} | Test Loss : {:.3f} '.format(test_acc, test_loss))
        torch.save(semisupervised_trained_model.state_dict(), 'saved_models/semi_supervised_weights')