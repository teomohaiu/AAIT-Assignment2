import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset.Dataset import ImageDataset, get_transforms
from trainer.train_pseudo_labelling import train_supervised, evaluate, train_semisupervised, evaluate_for_submission
from models.simple_convnet import ConvNet
from models.resnet50 import resnet50_model
import matplotlib.pyplot as plt
import yaml
import datetime
import pandas as pd

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--checkpoint_supervised_model', type=str, default=None, help='Checkpoint path for the supervised trained model')
    parser.add_argument('--checkpoint_semi_supervised_model', type=str, default=None, help='Checkpoint path for the semi-supervised trained model')
    parser.add_argument('--dataset', type=str, default='./task1/')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=100,  help='Number of training epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001,  help='Learning rate value')
    parser.add_argument('--momentum', type=float, default=0.9,  help='Momentum for SGD optimizer')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parser.add_argument('--problem_type', type=str, default='missing-labels', choices=['missing-labels', 'noisy-labels'], help='Wheter to use or not the unlabeled data')
    parser.add_argument('--loss_function', type=str, default='cross-entropy', choices=['cross-entropy', 'negative-loglikelihood'], help='Type of loss to use')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='Type of optimizer to use')
    args = parser.parse_args()

    return args



if __name__=='__main__':
    args = get_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == args.device, "The chosen device is not available!"

    train_imgs_dir = os.path.join(args.dataset, 'train_data/images')
    val_imgs_dir = os.path.join(args.dataset, 'val_data')
    annotations_file = os.path.join(args.dataset, 'train_data/annotations.csv')
    
    # Load datasets and create DataLoaders
    labeled_train_dataset = ImageDataset(annotations_file=annotations_file, img_dir=os.path.join(train_imgs_dir, 'labeled'), transform=get_transforms())
    if args.problem_type == 'missing-labels':
        unlabeled_train_dataset = ImageDataset(annotations_file=None, img_dir=os.path.join(train_imgs_dir, 'unlabeled'), transform=get_transforms())
        unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, args.batch_size)
    test_dataset = ImageDataset(annotations_file=None, img_dir=val_imgs_dir, transform=get_transforms())
    test_dataloader = DataLoader(test_dataset, args.batch_size)

    # Create logging directory and parameters file
    model_save_path = os.path.join("saved_models", args.model)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    experiment_dir =  os.path.join(model_save_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    with open(os.path.join(experiment_dir, 'hparams.yaml'), 'w') as file:
        yaml.dump(vars(args), file, default_flow_style=False)

    if args.model == 'simple_convnet':
        model = Net()
    elif args.model =="simple_convnet_2":
        model = ConvNet()
    elif args.model == "resnet50":
        model = resnet50_model()

    print(model)

    # Setup optimizers and loss function
    if args.loss_function == 'cross-entropy':
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_function == 'negative-loglikelihood':
        loss_fn = nn.NLLLoss()

    if args.optimizer == 'sgd':
        optimizer= optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters())

    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters: ', pytorch_total_params)

    if args.problem_type == 'missing-labels':
        train_size = int(0.8 * len(labeled_train_dataset))
        valid_size = len(labeled_train_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(labeled_train_dataset, [train_size, valid_size])
        labeled_train_dataloader = DataLoader(train_dataset, args.batch_size)
        labeled_valid_dataloader = DataLoader(valid_dataset, args.batch_size)
        # Perfom training from scratch if no model is provided
        if not args.checkpoint_supervised_model:
            # 1. Train the teacher network in a supervised manner
            trained_model = train_supervised(model, args.epochs, labeled_train_dataloader, labeled_valid_dataloader, loss_fn, optimizer,args.device)
        
            # Saved the trained model
            torch.save(trained_model.state_dict(), os.path.join(experiment_dir, 'supervised_weights'))

            # Load the model
            model.load_state_dict(torch.load(os.path.join(experiment_dir, 'supervised_weights')))
        else:
            model.load_state_dict(torch.load(args.checkpoint_supervised_model))


        # Perform evaluation on supervised model
        valid_acc, valid_loss = evaluate(model, labeled_valid_dataloader, loss_fn, args.device)
        print('Valid Acc : {:.5f} | Valid Loss : {:.3f} '.format(valid_acc, valid_loss))

        if not args.checkpoint_semi_supervised_model:
            # 2. Train the student model with respect to the weights learnt by the teacher model
            semisupervised_trained_model= train_semisupervised(model, 
                                                               args.epochs, 
                                                               labeled_train_dataloader, 
                                                               unlabeled_train_dataloader, 
                                                               labeled_valid_dataloader, 
                                                               loss_fn, 
                                                               optimizer,
                                                               args.device)
            torch.save(semisupervised_trained_model.state_dict(), os.path.join(experiment_dir, 'semi_supervised_weights'))
            model.load_state_dict(torch.load(os.path.join(experiment_dir, 'semi_supervised_weights')))
        else:
            model.load_state_dict(torch.load(args.checkpoint_semi_supervised_model))


        # Perform evaluation of the model after the help of teacher model
        valid_acc, valid_loss = evaluate(model, labeled_valid_dataloader, loss_fn, args.device)
        print('Valid Acc : {:.5f} | Valid Loss : {:.3f} '.format(valid_acc, valid_loss))
      
        # Perform prediction on test dataset and save the results
        img_paths, predictions = evaluate_for_submission(model, test_dataloader, device)
        results = {}
        results['sample'] = img_paths
        results['label'] = predictions
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(experiment_dir, 'submission.csv'))