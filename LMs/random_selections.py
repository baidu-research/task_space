"""
    Generate random selections, without repetition
"""
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_name_file')
    parser.add_argument('k', type=int, help='number of ckpt to pick')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to control the picks. Default: 0')
    parser.add_argument('--rounds', type=int, default=20,
                        help='simulate multiple independent picks')
    parser.add_argument('--save', default='random_k.txt',
                        help='path to a file that saves the picked ckpts')
    return parser.parse_args()


def pick(ckpt_name_file, k, rounds=20, save='random_k.txt', seed=0):
    np.random.seed(seed)
    all_ckpts = open(ckpt_name_file, 'r').read().strip().split()
    N = len(all_ckpts)
    if k == 1:
        assert rounds <= N
        ids = np.sort(np.random.choice(N, rounds, replace=False))
        with open(save, 'w') as f:
            for i in ids:
                f.write(all_ckpts[i]+'\n')
    else:
        all_ids = []
        f = open(save, 'w')
        for r in range(rounds):
            while True:
                ids = np.sort(np.random.choice(N, k, replace=False))
                # ensure any two rounds of picks are different
                if not any([np.array_equal(ids, _) for _ in all_ids]):
                    all_ids.append(ids)
                    f.write(' '.join([all_ckpts[i] for i in ids]) + '\n')
                    break


if __name__ == '__main__':
    args = parse_args()
    pick(args.ckpt_name_file, args.k, args.rounds, args.save, args.seed)
