import torch
import torch.nn.functional as F
import tqdm
from trainer.utils import alpha_weight
from torch.autograd import Variable

def evaluate(model, test_dataloader, device):
    model.eval()
    correct = 0 
    total = 0
    loss = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            predicted = torch.max(output,1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss += F.nll_loss(output, labels).item()

    return 100 * correct/total, (loss/len(test_loader))



def train_supervised(model, epochs, labeled_dataloader, test_dataloader, device):
    optimizer = torch.optim.SGD( model.parameters(), lr = 0.1)
    model.train()
    for epoch in range(epochs):
        correct = 0
        running_loss = 0
        total_train = 0
        train_accuracy = 0.0
        for batch_idx, (data, labels) in enumerate(labeled_dataloader):
            data = Variable(data.to(device))
            labels = Variable(labels.to(device))

            output = model(data)
            labeled_loss = F.nll_loss(output, labels)
                       
            optimizer.zero_grad()
            labeled_loss.backward()
            optimizer.step()
            running_loss += labeled_loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += labels.size(0)
            train_accuracy += (predicted == labels).sum().item()
        train_accuracy = train_accuracy / total_train
        
        #print('Epoch: {} : Train accuracy: {:.5f} | Train loss: {:.5f}'.format(epoch, train_accuracy, running_loss/len(labeled_dataloader)))
        # Evaluate every 10 epochs on test dataset
        if epoch % 10 == 0:
            test_acc, test_loss = evaluate(model, test_dataloader, device)
            print('Epoch: {} : Train Loss : {:.5f} | Test Acc : {:.5f} | Test Loss : {:.3f} '.format(epoch, running_loss/(10 *len(labeled_dataloader)), test_acc, test_loss))
            model.train()
    
    return model




def train_semisupervised(model, epochs, train_loader, unlabeled_loader, test_loader, device):
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

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
            unlabeled_loss = alpha_weight(step) * F.nll_loss(output, pseudo_labeled)   
            
            # Backpropogate
            optimizer.zero_grad()
            unlabeled_loss.backward()
            optimizer.step()
            
            
            # For every 10 batches train one epoch on labeled data 
            if batch_idx % 10 == 0:
                
                # Normal training procedure
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    output = model(X_batch)
                    labeled_loss = F.nll_loss(output, y_batch)

                    optimizer.zero_grad()
                    labeled_loss.backward()
                    optimizer.step()
                
                # Now we increment step by 1
                step += 1
                

        test_acc, test_loss =evaluate(model, test_loader)
        print('Epoch: {} : Alpha Weight : {:.5f} | Test Acc : {:.5f} | Test Loss : {:.3f} '.format(epoch, alpha_weight(step), test_acc, test_loss))
        
        """ LOGGING VALUES """
        alpha_log.append(alpha_weight(step))
        test_acc_log.append(test_acc/100)
        test_loss_log.append(test_loss)
        """ ************** """
        model.train()


    return model

