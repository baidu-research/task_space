"""
    Plot accuracy:
        python analyze_result.py $path_to_MMI $path_to_peek $path_to_random -l MMI peek random

    Plot gain over random:
        python analyze_result.py $path_to_MMI $path_to_peek $path_to_random -l MMI peek random --gain-over-random
"""
import argparse
import glob
import numpy as np
import os
import pylab
import subprocess
pylab.rcParams['font.size']=20


colors=['r', 'g', 'b', 'c', 'm', 'y', 'k']
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dirs', nargs='+', help='one or multiple dirs, '
                        'each contains logs that corresponds to a '
                        'ckpt selection method')
    parser.add_argument('-l', '--label', nargs='+',
                        help='label for each dir')
    parser.add_argument('--gain-over-random', action='store_true',
                        help='plot the gain of each non-random method '
                        'against random')
    parser.add_argument('-s', help='path to save the figure. If not given, '
                        'will show the figure')
    return parser.parse_args()


def grep_result(log_dir):
    """
        Return a numpy array of validation accuracies for all log files under
        the dir. The logs are due to a certain selection methods of checkpoints.
        They are named as selection_k${k_ckpts}_t${task_id}
        
        Suppose there are T tasks, and we select up to K checkpoints. 

        If random selection with R different random seeds, the returned is
            R x T x K array

        If other (determnistic) method, the returned is
            T x K array
    """
    fs = glob.glob(os.path.join(log_dir, '*_k[1-9]*_t[0-9]*.out'))
    fs = [f.split('/')[-1] for f in fs]
    K = max([int(f.split('.')[0].split('_')[1][1:]) for f in fs])
    T = max([int(f.split('.')[0].split('_')[2][1:]) for f in fs]) + 1
    is_random_selection = 'seed' in fs[0]
    if is_random_selection:
        seeds = [int(f.split('_')[0][4:]) for f in fs]
        seeds = np.unique(seeds)
        R = len(seeds)
        Results = np.zeros((R, T, K))
        for i, s in enumerate(seeds):
            for t in range(T):
                for k in range(1, K+1):
                    fname = os.path.join(log_dir,
                                         'seed{}_k{}_t{}.out'.format(s, k, t))
                    acc = float(
                            subprocess.check_output(
                                ['./get_acc.sh', fname]).decode('utf-8')\
                                .strip()) * 100
                    Results[i, t, k-1] = acc
    else:
        method=fs[0].split('_')[0]
        Results = np.zeros((T, K))
        for t in range(T):
            for k in range(1, K+1):
                fname = os.path.join(log_dir,
                                     '{}_k{}_t{}.out'.format(method, k, t))
                acc = float(
                        subprocess.check_output(['./get_acc.sh', fname])\
                        .decode('utf-8').strip()) * 100
                Results[t, k-1] = acc
    return Results


def plot_results(Results, label, color):
    if len(Results.shape) == 3:
        # random baseline
        R, T, K = Results.shape
        task_average = np.mean(Results, axis=1)
        q_lower = np.quantile(task_average, 0.05, axis=0)
        q_upper = np.quantile(task_average, 0.95, axis=0)
        pylab.fill_between(range(1, K+1), q_lower, q_upper,
                           alpha=0.2, label=label, color=color)
    else: # deterministic selection method
        T, K = Results.shape
        task_average = np.mean(Results, axis=0)
        pylab.plot(range(1, K+1), task_average, label=label, color=color)


def plot_gains(result, random_result, label, color):
    random_mean = random_result.mean(axis=0) # marginalize over random seed
    gain_each_task = result - random_mean
    gain = gain_each_task.mean(axis=0)
    pylab.plot(range(1, len(gain)+1), gain, label=label, color=color)


if __name__ == '__main__':
    args = parse_args()
    if args.label:
        assert len(args.label) == len(args.log_dirs)
        label = args.label
    else:
        label = [str(_) for _ in range(1, len(args.log_dirs)+1)]

    if args.gain_over_random:
        Results = []
        for i, d in enumerate(args.log_dirs):
            Result = grep_result(d)
            if len(Result.shape) == 3:
                random_index = i
                random_result = Result
            else:
                Results.append(Result)
        for i, result in enumerate(Results):
            plot_gains(result, random_result, label[i], colors[i])
        pylab.hlines(0, 1, random_result.shape[-1],
                     colors[random_index], 'dashed', label='random')
        #pylab.ylabel(r'$\mathbb{E}_t[acc_t - acc_t^r]$ (%)')
        pylab.ylabel('gain over random (%)')
    else:
        for i, d in enumerate(args.log_dirs):
            Results = grep_result(d)
            plot_results(Results, label[i], colors[i])
            pylab.ylabel(r'$\mathbb{E}_t acc_t$ (%)')
    pylab.xlabel('k checkpoints used')
    pylab.xticks(range(0, 21, 5), [str(_) for _ in range(0, 21, 5)])
    pylab.legend()
    pylab.tight_layout()
    if args.s:
        pylab.savefig(args.s)
    else:
        pylab.show()
