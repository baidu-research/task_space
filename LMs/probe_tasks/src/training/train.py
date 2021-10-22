import torch
from collections import OrderedDict
from .utils import use_gpu


def train_one_epoch(model, batches, optimizer, **kwargs):
    """
        Train a task-specific model for one epoch.
        
        Input:
        model (nn.Module): The model could include some or all leanable
            parameters in a pretrained feature extractor.
        batches (iterable): iterate over all batches in an epoch
        kwargs:
            epoch_id (int): start from 0, epoch id
            total_iter (int): total number of iterations so far

        Return:
        iters: total number of iterations so far
    """
    model.train()
    epoch = kwargs.get('epoch_id', 0)
    iters = kwargs.get('total_iter', 0)
    for inputs, targets, mask in batches:
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()
            mask = mask.cuda()
        optimizer.zero_grad()
        predict, loss_i = model(inputs, targets, mask)
        print("Epoch {} Iter {} | train_loss {:.5f}".format(
              epoch, iters, loss_i.item()), flush=True)
        loss_i.backward()
        optimizer.step()
        iters += 1
    return iters


@torch.no_grad()
def eval_model(model, batches, metrics):
    """
        Evaluate the model. Return loss and metrics
    """
    model.eval()
    total_samples = 0
    total_loss = 0.
    for inputs, targets, mask in batches:
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()
            mask = mask.cuda()
        num_non_pad = mask.sum()
        total_samples += num_non_pad
        predict, loss = model(inputs, targets, mask)
        total_loss += loss.item() * num_non_pad
        
        for metric_name in metrics:
            metrics[metric_name](predict, targets, mask)

    reports = OrderedDict({'loss': total_loss / total_samples})
    for metric_name in metrics:
        reports[metric_name] = metrics[metric_name].report()
    return reports
