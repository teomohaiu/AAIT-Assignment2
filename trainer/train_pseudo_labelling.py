import torch
import torch.nn.functional as F
import tqdm
from trainer.utils import alpha_weight
from torch.autograd import Variable
import os

def evaluate(model, test_dataloader, loss_fn, device):
    model.eval()
    correct = 0 
    total = 0
    loss = 0
    with torch.no_grad():
        for data, labels in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            predicted = torch.max(output,1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss += loss_fn(output, labels).item()

    return correct/total, (loss/len(test_dataloader))



def evaluate_for_submission(model, test_dataloader, device):
    model.eval()
    img_paths = list()
    predictions = list()
    with torch.no_grad():
        for data, img_path in test_dataloader:
            data = data.to(device)
            output = model(data)
            predicted = torch.max(output,1)[1]
            filenames = [f'img{os.path.basename(path)}' for path in img_path]
            preds = [p.item() for p in predicted]
            img_paths.extend(filenames)
            predictions.extend(preds)

    return img_paths, predictions



def train_supervised(model, epochs, train_dataloader, valid_dataloader, loss_fn, optimizer, device):
    model.train()
    for epoch in range(epochs):
        correct = 0
        running_loss = 0
        total_train = 0
        train_accuracy = 0.0
        for batch_idx, (data, labels) in enumerate(train_dataloader):
            data = Variable(data.to(device))
            labels = Variable(labels.to(device))

            output = model(data)
            labeled_loss = loss_fn(output, labels)
                       
            optimizer.zero_grad()
            labeled_loss.backward()
            optimizer.step()
            running_loss += labeled_loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += labels.size(0)
            train_accuracy += (predicted == labels).sum().item()
        train_accuracy = train_accuracy / total_train
        
        print('Epoch: {} : Train accuracy: {:.5f} | Train loss: {:.5f}'.format(epoch, train_accuracy, running_loss/len(train_dataloader)))
        # Evaluate every 10 epochs on valid dataset
        if epoch % 10 == 0:
            valid_acc, valid_loss = evaluate(model, valid_dataloader,  loss_fn, device)
            print('Epoch: {} : Train Loss : {:.5f} | Valid Acc : {:.5f} | Valid Loss : {:.3f} '.format(epoch, running_loss/(10 *len(train_dataloader)), valid_acc, valid_loss))
            model.train()
    
    return model



def train_semisupervised(model, epochs, train_loader, unlabeled_loader, valid_loader, loss_fn, optimizer, device):
    # Instead of using current epoch we use a "step" variable to calculate alpha_weight
    # This helps the model converge faster
    step = 100 

    alpha_log = []
    test_acc_log = []
    test_loss_log = []
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, x_unlabeled in enumerate(unlabeled_loader):
            
            
            # Forward Pass to get the pseudo labels
            x_unlabeled = x_unlabeled[0].to(device)
            model.eval()
            output_unlabeled = model(x_unlabeled)
            _, pseudo_labeled = torch.max(output_unlabeled, 1)
            model.train()
            
            
            # Now calculate the unlabeled loss using the pseudo label
            output = model(x_unlabeled)
            unlabeled_loss = alpha_weight(step) * loss_fn(output, pseudo_labeled)   
            
            # Backpropogate
            optimizer.zero_grad()
            unlabeled_loss.backward()
            optimizer.step()
            
                   
         # Normal training procedure
        for X_batch, y_batch in train_loader:
            X_batch = Variable(X_batch.to(device))
            y_batch = Variable(y_batch.to(device))
            output = model(X_batch)
            labeled_loss = loss_fn(output, y_batch)

            optimizer.zero_grad()
            labeled_loss.backward()
            optimizer.step()
                
        # Now we increment step by 1
        step += 1     

        valid_acc, valid_loss = evaluate(model, valid_loader, loss_fn, device)
        print('Epoch: {} : Alpha Weight : {:.5f} | Valid Acc : {:.5f} | Valid Loss : {:.3f} '.format(epoch, alpha_weight(step), valid_acc, valid_loss))
        
        """ LOGGING VALUES """
        alpha_log.append(alpha_weight(step))
        test_acc_log.append(valid_acc/100)
        test_loss_log.append(valid_loss)
        """ ************** """
        model.train()


    return model

