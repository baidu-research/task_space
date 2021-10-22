"""
    Admin extracts training and validation features for its new tasks,
    using an agent checkpoint

    For now, we don't back-prop for any new task, i.e., the classification head
    is the only trainable unit
"""
import argparse
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from utils import use_gpu, recast_labels, \
                load_ckpt_for_feature_extraction, extract


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_dir', type=str)
    parser.add_argument('split', type=str,
                        help='path to an .npz of admin and agent data ids')
    parser.add_argument('save', type=str,
                        help='path to save the features.')
    return parser.parse_args()


def admin_trainTest_dataLoader(admin_train_ids, admin_test_ids, mean_std,
                               batch_size=128, shuffle=False):
    """
        We don't use data augmentation for simplicity
    """
    transform_data = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_std[0], mean_std[1])
    ])
    train = torchvision.datasets.CIFAR100('images', True, transform_data)
    test = torchvision.datasets.CIFAR100('images', False, transform_data)

    # "normalize" train and test ids
    uniq_train_ids, train_id_map = np.unique(admin_train_ids, return_inverse=True)
    train_id_map = np.reshape(train_id_map, admin_train_ids.shape)

    uniq_test_ids, test_id_map = np.unique(admin_test_ids, return_inverse=True)
    test_id_map = np.reshape(test_id_map, admin_test_ids.shape)

    # train data loader
    train.data = train.data[uniq_train_ids]
    train.targets = recast_labels(np.array(train.targets)[uniq_train_ids])
    train_loader = DataLoader(train, batch_size, shuffle=shuffle)

    # test data loader
    test.data = test.data[uniq_test_ids]
    test.targets = recast_labels(np.array(test.targets)[uniq_test_ids])
    test_loader = DataLoader(test, batch_size, shuffle=shuffle)

    return train_loader, test_loader, train_id_map, test_id_map


if __name__ == '__main__':
    args = parse_args()
    split_file = np.load(args.split)
    admin_train_ids = split_file['admin_train_ids']
    admin_test_ids = split_file['admin_test_ids']
    
    mean_std = np.load(os.path.join(args.ckpt_dir, 'mean_std.npz'))
    mean_std = (mean_std['mean'], mean_std['std'])
    train_loader, test_loader, train_id_map, test_id_map = \
        admin_trainTest_dataLoader(admin_train_ids, admin_test_ids, mean_std)
    net = load_ckpt_for_feature_extraction(args.ckpt_dir)
    X_train, y_train = extract(net, train_loader)
    X_test, y_test = extract(net, test_loader)
    save_under_dir = os.path.dirname(os.path.abspath(args.save))
    if not os.path.exists(save_under_dir):
        os.makedirs(save_under_dir)
    np.savez(args.save,
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test,
             train_id_map=train_id_map, test_id_map=test_id_map)
