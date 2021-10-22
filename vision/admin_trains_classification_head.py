"""
    Admin uses/concates feature(s) from one/multiple agent feature extractors,
    and trains a softmax classifier using task-specific training data
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


use_gpu = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('feat_files', nargs='+',
                        help='path to one or multiple features (.npz file) '
                        'by one or multiple models')
    parser.add_argument('--task-id', type=int, default=0,
                        help='which admin task to run')
    parser.add_argument('-e', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('-b', type=int, default=128,
                        help='batch size')
    parser.add_argument('-lr', type=float, default=1e-1,
                        help='learning rate')
    return parser.parse_args()


def recast_labels(labels):
    """
        recast class labels to [0, n_class-1]
    """
    _, recasted_labels = np.unique(labels, return_inverse=True)
    return recasted_labels.tolist()


def read_feats(feat_files, task_id):
    X_train = []
    X_test = []
    for f in feat_files:
        read_f = np.load(f)
        X_train.append(read_f['X_train'][read_f['train_id_map'][task_id]])
        X_test.append(read_f['X_test'][read_f['test_id_map'][task_id]])
    X_train = np.hstack(X_train)
    X_test = np.hstack(X_test)
    y_train = recast_labels(read_f['y_train'][read_f['train_id_map'][task_id]])
    y_test = recast_labels(read_f['y_test'][read_f['test_id_map'][task_id]])
    return X_train, y_train, X_test, y_test


def build_model(feat_dim, num_class):
    """
        Build MLP classifier
    """
    model = nn.Sequential(
                nn.Linear(feat_dim, num_class),
                )
    if use_gpu:
        model.cuda()
    return model


def train_one_epoch(epoch):
    model.train()
    n_samples = X_train.shape[0]
    iter_within_epoch = 0
    for i in range(0, n_samples, args.b):
        ids = np.random.permutation(n_samples)
        X_i = X_train[ids[i:min(i+args.b, n_samples)]]
        y_i = y_train[ids[i:min(i+args.b, n_samples)]]
        if use_gpu:
            X_i = X_i.cuda()
            y_i = y_i.cuda()

        optimizer.zero_grad()
        outputs = model(X_i)
        loss_i = loss(outputs, y_i)
        loss_i.backward()
        optimizer.step()

        iter_within_epoch += 1
        iters = epoch * int(np.ceil(n_samples/args.b)) + iter_within_epoch
        print('Training Epoch: {}, Iter: {}, Loss: {:.4f}'.format(
              epoch, iters, loss_i), flush=True)


@torch.no_grad()
def eval_model(epoch):
    model.eval()
    n_samples = X_test.shape[0]

    test_loss = 0.0
    test_acc = 0.0
    for i in range(0, n_samples, args.b):
        X_i = X_test[i: min(i+args.b, n_samples)]
        y_i = y_test[i: min(i+args.b, n_samples)]
        if use_gpu:
            X_i = X_i.cuda()
            y_i = y_i.cuda()
        outputs = model(X_i)
        loss_i = loss(outputs, y_i)
        _, preds = outputs.max(1)
        
        test_loss += loss_i.item() * len(y_i) 
        test_acc += preds.eq(y_i).sum()

    test_loss /= n_samples
    test_acc /= n_samples
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}'.format(
          epoch, test_loss, test_acc), flush=True)
    return test_loss, test_acc


if __name__ == '__main__':
    args = parse_args()
    print("For task {}, using checkpoints:".format(args.task_id), flush=True)
    for f_file in args.feat_files:
        print(f_file, flush=True)
    print("Traing with lr={}, batch={}, for {} epochs".format(
          args.lr, args.b, args.e), flush=True)
    X_train, y_train, X_test, y_test = read_feats(args.feat_files, args.task_id)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    model = build_model(X_train.shape[1], len(torch.unique(y_train)))
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1)

    best_acc = 0.0
    for e in range(args.e):
        train_one_epoch(e)
        test_loss, test_acc = eval_model(e)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = e
        scheduler.step()

    print("Best test accuracy {:.4f}, achieved at epoch {}".format(
          best_acc, best_epoch), flush=True)
