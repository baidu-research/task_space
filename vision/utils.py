import os
import sys
import re
import datetime

import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler

use_gpu = torch.cuda.is_available()

def get_network(net_name, classes, iseed):
    """
        Return a network of specified type
    """
    torch.manual_seed(iseed)  # random seed for initialization
    if net_name == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(classes)
    elif net_name == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(classes)
    elif net_name == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(classes)
    elif net_name == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(classes)
    elif net_name == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(classes)
    elif net_name == 'densenet161':
        from models.densenet import densenet161
        net = densenet161(classes)
    elif net_name == 'densenet169':
        from models.densenet import densenet169
        net = densenet169(classes)
    elif net_name == 'densenet201':
        from models.densenet import densenet201
        net = densenet201(classes)
    elif net_name == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet(classes)
    elif net_name == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3(classes)
    elif net_name == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4(classes)
    elif net_name == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2(classes)
    elif net_name == 'xception':
        from models.xception import xception
        net = xception(classes)
    elif net_name == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(classes)
    elif net_name == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(classes)
    elif net_name == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(classes)
    elif net_name == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(classes)
    elif net_name == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(classes)
    elif net_name == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18(classes)
    elif net_name == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34(classes)
    elif net_name == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50(classes)
    elif net_name == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101(classes)
    elif net_name == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152(classes)
    elif net_name == 'resnext50':
        from models.resnext import resnext50
        net = resnext50(classes)
    elif net_name == 'resnext101':
        from models.resnext import resnext101
        net = resnext101(classes)
    elif net_name == 'resnext152':
        from models.resnext import resnext152
        net = resnext152(classes)
    elif net_name == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet(classes)
    elif net_name == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2(classes)
    elif net_name == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet(classes)
    elif net_name == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet(classes)
    elif net_name == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(classes)
    elif net_name == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet(classes)
    elif net_name == 'attention56':
        from models.attention import attention56
        net = attention56(classes)
    elif net_name == 'attention92':
        from models.attention import attention92
        net = attention92(classes)
    elif net_name == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18(classes)
    elif net_name == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34(classes)
    elif net_name == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50(classes)
    elif net_name == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101(classes)
    elif net_name == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152(classes)
    elif net_name == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet(classes)
    elif net_name == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18(classes)
    elif net_name == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34(classes)
    elif net_name == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50(classes)
    elif net_name == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101(classes)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]


def recast_labels(labels):
    """
        recast class labels to [0, n_class-1]
    """
    _, recasted_labels = np.unique(labels, return_inverse=True)
    return recasted_labels.tolist()


def load_ckpt_for_feature_extraction(ckpt_dir):
    """
        Load a checkpoint for feature extraction. The last fc layer (softmax 
        classifier) is often futile. So num_class is a dummy variable
    """
    net_type = open(os.path.join(ckpt_dir, 'info')).readline().strip()
    net = get_network(net_type, 10, 0)
    
    agent_model_states = torch.load(os.path.join(ckpt_dir, 'ckpt.pt'),
                                    map_location='cpu')
    agent_model_states.pop('fc.weight')
    agent_model_states.pop('fc.bias')

    net_state_dict = net.state_dict()
    net_state_dict.update(agent_model_states)
    net.load_state_dict(net_state_dict)
    if use_gpu:
        net.cuda()
    return net 


@torch.no_grad()
def extract(net, data_loader):
    net.eval()
    feats = []
    labels = []
    for i, img_and_label in enumerate(data_loader):
        if isinstance(img_and_label, list):
            img, label = img_and_label[0], img_and_label[1]
            labels.append(label)
        else:  # just image
            img = img_and_label
            label = None

        if use_gpu:
            img = img.cuda()
        outputs = net.extract(img)
        feats.append(outputs.data.cpu().numpy())
        print(i)

    feats = np.vstack(feats)
    if len(labels) > 0:
        labels = np.hstack(labels)

    return feats, labels
