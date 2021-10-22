"""
    Admin wants to work on the its classification problem, but it only has a
    small number (e.g., 20) of training data points for each class.
    
    There are K agents. Each of them sees S classes of training data. Note that
    their training data doesn't overlap with that of admin.
    
    This file simulates the agents. It reads in the classes (tasks) seen by each
    agent, and launch a job for it.
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader 

from conf import settings
from utils import use_gpu, get_network, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, \
    recast_labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('split', type=str,
                        help='path to an .npz of admin and agent data ids')
    parser.add_argument('agent_id', type=int, help='agent id')
    parser.add_argument('--net', type=str, default='resnet50',
                        help='net type, default to resnet50')
    
    # optimization parameters
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-e', type=int, default=settings.EPOCH, help='number of epochs')
    parser.add_argument('-s', '--save_dir', type=str, default='',
                        help='path to checkpoint model and features. The '
                        'dir is created under ./checkpoint/')
    args = parser.parse_args()
    return args


def create_agent_dataLoaders(agent_train_ids, agent_test_ids, batch_size=16):
    # get mean and std of the subset
    all_train = torchvision.datasets.CIFAR100('images', train=True)
    mean = [(all_train.data[agent_train_ids, :, :, _]/255).mean() for _ in range(3)]
    std = [(all_train.data[agent_train_ids, :, :, _]/255).std() for _ in range(3)]

    # train data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train = torchvision.datasets.CIFAR100('images', True, transform_train)
    train.data = train.data[agent_train_ids]
    train.targets = recast_labels(np.array(train.targets)[agent_train_ids])
    train_loader = DataLoader(train, batch_size, shuffle=True, num_workers=2)

    # test data loader
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test = torchvision.datasets.CIFAR100('images', False, transform_test)
    test.data = test.data[agent_test_ids]
    test.targets = recast_labels(np.array(test.targets)[agent_test_ids])
    test_loader = DataLoader(test, batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, (mean, std)


def train_agent_one_epoch(epoch):
    net.train()

    for batch_index, (images, labels) in enumerate(train_loader):
        if use_gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLR: {:0.6f}\tLoss: {:0.4f}'.format(
            optimizer.param_groups[0]['lr'],
            loss.item(),
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(train_loader.dataset)
        ), flush=True)

        if epoch <= args.warm:
            warmup_scheduler.step()


@torch.no_grad()
def eval_agent(epoch):
    net.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    test_loss /= len(test_loader.dataset)
    acc = correct.float() / len(test_loader.dataset)
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}'.format(
          epoch, test_loss, acc), flush=True)

    return test_loss, acc

    
if __name__ == '__main__':
    args = parse_args()
    split_file = np.load(args.split)
    agent_train_ids = split_file['agent_train_ids'][args.agent_id]
    agent_test_ids = split_file['agent_test_ids'][args.agent_id]
    num_classes = len(split_file['agent_classes'][args.agent_id])
    net = get_network(args.net, num_classes, 0)
    if use_gpu:
        net.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=settings.MILESTONES, gamma=0.2) #learning rate decay

    train_loader, test_loader, mean_std = create_agent_dataLoaders(
                                            agent_train_ids,
                                            agent_test_ids,
                                            args.b)
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.save_dir)
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    np.savez(os.path.join(checkpoint_path, 'mean_std.npz'),
             mean=mean_std[0], std=mean_std[1])
    open(os.path.join(checkpoint_path, 'info'), 'w').write(args.net)

    best_acc = 0.0
    best_epoch = -1

    for epoch in range(1, args.e + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train_agent_one_epoch(epoch)
        test_loss, acc = eval_agent(epoch)

        if best_acc < acc:
            weights_path = os.path.join(checkpoint_path, 'ckpt.pt')
            print('saving weights file to {}'.format(weights_path), flush=True)
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            best_epoch = epoch

    print("Test acc best: {} at epoch {}".format(best_acc, best_epoch),
          flush=True)
