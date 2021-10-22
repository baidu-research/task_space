import argparse
import glob
import numpy as np
import subprocess
import os
from pathlib import Path
import pylab
import matplotlib.patches as mpatches
pylab.rcParams['font.size']=20

K = 5  # max number of agents
dot_type = ['d', 'o', '*', 'h', 't']
quantile = 0.85

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('MI_log_path', type=str,
                        help='path to MI logs such that '
                        '$MI_log_path/k[1-K].log are the logs')
    parser.add_argument('random_log_path', type=str,
                        help='path to random selection logs such that '
                        '$random_log_path/k[1-K]_s${seed}.log are the logs')
    parser.add_argument('-e', '--exclude-single-model-id', type=int, nargs='+',
                        help='Exclude this model for random picks of single '
                        'model. Usually exclude the first pick by MI, if it '
                        'appears in one of the K=1 random picks')
    parser.add_argument('-m', '--metric-name', type=str, default='acc_k=1',
                        help='name of the metric. Default to top-1 accuracy')
    parser.add_argument('-si', '--single-model-id', type=int, nargs='+',
                        help='ID (start from 0) of the single '
                        'model, whose result is a baseline. Default to 5 '
                        '(bert-base-uncased) and 15 (bert-large-cased)')
    parser.add_argument('-sm', '--single-model-name', type=str, nargs='+',
                        help='name of the single model baseline, default to '
                        'bert-base-uncased and bert-large-cased')
    parser.add_argument('--no-legend', action='store_true',
                        help='default to add legend')
    parser.add_argument('-y', nargs=2, type=float,
                        help='y-axis range')
    parser.add_argument('-s', type=str, help='name to save figure')
    return parser.parse_args()


def get_accuracy_for_single_model(random_log_path, metric_name, ckpt_id):
    log = os.path.join(random_log_path, 'k1_s{}.log'.format(ckpt_id))
    metric = float(subprocess.check_output(
                ['./get_best_test_metric.sh', log, metric_name])\
                .decode('utf-8').strip().split()[-1])
    return metric


def get_accuracy_for_random(random_log_path, metric_name, exclude_id=None):
    metrics = []
    for k in range(1, K+1):
        logs = glob.glob('{}/k{}_s*.log'.format(random_log_path, k))
        # exclude for single-model picks
        if k == 1 and exclude_id is not None:
            logs = [log for log in logs \
                    if int(Path(log).stem.split('_')[1][1:]) not in exclude_id]

        metric_k = []
        for log in logs:
            metric = subprocess.check_output(
                        ['./get_best_test_metric.sh', log, metric_name])\
                        .decode('utf-8').strip().split()[-1]
            metric_k.append(float(metric))
        metrics.append(metric_k)
    return metrics


def get_acuracy_for_MI(MIorE_log_path, metric_name):
    metrics = np.zeros(K)
    for k in range(1, K+1):
        log = '{}/k{}.log'.format(MIorE_log_path, k)
        metrics[k-1] = float(subprocess.check_output(
                        ['./get_best_test_metric.sh', log, metric_name])\
                        .decode('utf-8').strip().split()[-1])
    return metrics


if __name__ == '__main__':
    args = parse_args()
    random_results = get_accuracy_for_random(
            args.random_log_path, args.metric_name, args.exclude_single_model_id)
    MI_results = get_acuracy_for_MI(args.MI_log_path, args.metric_name)
    pylab.plot(range(1, K+1), MI_results, 'o-', color='r', linewidth=2)
    
    if args.single_model_id is not None:
        for i, s_m_id in enumerate(args.single_model_id):
            single_baseline = get_accuracy_for_single_model(
                                    args.random_log_path, args.metric_name, s_m_id)
            pylab.scatter(1, single_baseline, s=100, marker=dot_type[i],
                          facecolors='none', edgecolors='g', linewidths=2,
                          label=args.single_model_name[i])
    vplot = pylab.violinplot(random_results, showmedians=True)

    if not args.no_legend:
        labels = ['MMI']
        if args.single_model_name:
            labels += args.single_model_name
        labels.append('random')
        pylab.legend(labels,
                     loc='lower right', fancybox=True, framealpha=0.5)
    pylab.xlabel(r'$K$' + ' checkpoints used')
    if args.metric_name:
        pylab.ylabel(args.metric_name + ' (%)')
    pylab.tight_layout()
    if args.y:
        pylab.ylim(args.y)
    if args.s:
        pylab.savefig(args.s)
    else:
        pylab.show()
