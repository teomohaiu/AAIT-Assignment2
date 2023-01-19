import torch
import os
import shutil
import numpy as np
from sklearn.metrics import top_k_accuracy_score, accuracy_score

def train_MixMatch_one_epoch(model, labeled_trainloader,  unlabeled_trainloader, epoch, train_iteration,  criterion, optimizer, ema_optimizer, T, alpha, device, nr_epochs):
     labeled_train_iter = iter(labeled_trainloader)
     unlabeled_train_iter = iter(unlabeled_trainloader)

     losses = list()
     losses_x = list()
     losses_u = list()

     model.train()
     for batch_idx in range(train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)

        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter), next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_, inputs_u2 = next(unlabeled_train_iter), next(unlabeled_train_iter)
            
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 100).scatter_(1, targets_x.view(-1,1).long(), 1)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u[0].cuda()
        inputs_u2 = inputs_u2[0].cuda()


        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(alpha, alpha)

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], lambda_u=75, epoch= epoch+batch_idx/train_iteration, nr_epochs=nr_epochs)

        loss = Lx + w * Lu

        # record loss
        losses.append(loss.item())
        losses_x.append(Lx.item())
        losses_u.append(Lu.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()


     return torch.Tensor(losses).sum()/ len(losses), torch.Tensor(losses_x).sum()/ len(losses_x), torch.Tensor(losses_u).sum()/len(losses_u)


def validate_one_epoch(valloader, model, criterion, accuracy, device):
    losses = list()
    accuracies = list()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            y_pred = [o.argmax().cpu().numpy() for o in outputs]
            accuracy = accuracy_score(np.array(y_pred), targets.cpu().numpy())
            losses.append(loss.item())
            accuracies.append(accuracy)

          
    return torch.Tensor(losses).sum()/len(losses), torch.Tensor(accuracies).sum()/len(accuracies)

def train_MixMatch(model, ema_model, epochs, labeled_trainloader,unlabeled_trainloader, val_loader, test_loader, train_iterations, optimizer,ema_optimizer, train_criterion, criterion, T, alpha, device, experiment_dir):
    step = 0
    best_acc = 0 
    # Train and val
    for epoch in range(epochs):

        print('\nEpoch: [%d | %d]' % (epoch + 1, epochs))
        train_loss, train_loss_x, train_loss_u = train_MixMatch_one_epoch(model, labeled_trainloader,unlabeled_trainloader,epoch,train_iterations,train_criterion,optimizer, ema_optimizer, T, alpha, device, epochs)
        eval_train_loss, train_acc = validate_one_epoch(labeled_trainloader, ema_model, criterion, accuracy, device)
        val_loss, val_acc = validate_one_epoch(val_loader, ema_model, criterion, accuracy, device)

        step = train_iterations * (epoch + 1)

        print(f'Train accuracy: {train_acc} | train loss: {eval_train_loss} | valid accuracy: {val_acc} | valid loss: {val_loss}')

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, experiment_dir)
        


    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets



def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]