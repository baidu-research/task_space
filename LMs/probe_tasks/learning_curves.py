import argparse
import numpy as np
import pylab
import subprocess


colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', nargs='+')
    parser.add_argument('-w', '--smooth_window', type=int, default=1,
                        help='a moving rectangular window to smooth the loss')
    parser.add_argument('-m', '--metric', default='acc_k=1',
                        help='metric to plot, default to "acc_k=1", top-1 '
                        'accuracy')
    parser.add_argument('-e', '--epoch', type=int,  help='max epochs to show')
    parser.add_argument('-l', '--label', nargs='+')
    return parser.parse_args()


def get_train_loss(log, smooth_window):
    loss = subprocess.check_output(
                ['./utils/get_train_loss.sh', log]).decode('utf-8')
    loss = loss.strip().split()
    loss = [float(_) for _ in loss]
    if smooth_window == 1:
        return loss
    else:
        return np.convolve(loss, np.ones(smooth_window), 'valid') / smooth_window


def get_epoch_size(log):
    epoch_size = int(subprocess.check_output(
                ['./utils/get_epoch_size.sh', log]).decode('utf-8'))
    return epoch_size


def get_dev_or_test_loss(log, tag='dev'):
    loss = subprocess.check_output(
                ['./utils/get_dev_or_test_loss.sh', log, tag]).decode('utf-8')
    loss = loss.strip().split()
    loss = [float(_) for _ in loss]
    return loss


def get_dev_or_test_metric(log, tag='dev', metric='acc_k=1'):
    metric = subprocess.check_output(
                ['./utils/get_dev_or_test_metric.sh', log, tag, metric]).decode('utf-8')
    metric = metric.strip().split()
    metric = [float(_) for _ in metric]
    return metric


def plot_loss(train_loss, dev_loss, test_loss, epoch_size, label, color, epoch):
    pylab.plot(train_loss, color+':')
    epochs = len(dev_loss)
    pylab.plot(range(0, epochs*epoch_size, epoch_size), dev_loss, color+'--')
    pylab.plot(range(0, epochs*epoch_size, epoch_size), test_loss, color)
    pylab.xlim(left=0)
    if epoch:
        pylab.xlim(right=epoch*epoch_size)

def plot_metric(dev_metric, test_metric, epoch_size, label, color, epoch):
    epochs = len(dev_metric)
    pylab.plot(range(epoch_size, epochs*epoch_size, epoch_size), dev_metric[1:],
               color+'--', label=label+': dev')
    pylab.plot(range(epoch_size, epochs*epoch_size, epoch_size), test_metric[1:],
               color, label=label+': test')
    pylab.xlim(left=0)
    if epoch:
        pylab.xlim(right=epoch*epoch_size)


if __name__ == '__main__':
    args = parse_args()
    if not args.label:
        labels = [str(_) for _ in range(1, len(args.log_files)+1)]
    else:
        labels = args.label

    fig = pylab.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    pylab.ylabel('loss')
    ax2 = fig.add_subplot(2, 1, 2)
    pylab.ylabel(args.metric)
    for i, log in enumerate(args.log_files):
        train_loss = get_train_loss(log, args.smooth_window)
        dev_loss = get_dev_or_test_loss(log, tag='dev')
        test_loss = get_dev_or_test_loss(log, tag='test')
        epoch_size = get_epoch_size(log)

        pylab.sca(ax1)
        plot_loss(train_loss, dev_loss, test_loss, epoch_size, labels[i], colors[i], args.epoch)

        dev_metric = get_dev_or_test_metric(log, tag='dev', metric=args.metric) 
        test_metric = get_dev_or_test_metric(log, tag='test', metric=args.metric)
        pylab.sca(ax2)
        plot_metric(dev_metric, test_metric, epoch_size, labels[i], colors[i], args.epoch)

    pylab.legend()
    pylab.show()
