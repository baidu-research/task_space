"""
    Admin load feat files (.npz one for each model) and compute model similarity
"""
import argparse
import numpy as np
import os
import sys
from numpy.linalg import norm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('feat_files', nargs='+',
                        help='path to one or multiple features (.npz file) '
                        'by one or multiple models')
    parser.add_argument('--save', required=True, type=str,
                        help='name of or path to a .npy of similarity matrix, '
                        'and a .txt of a list of feature files')
    return parser.parse_args()


def KA(feat1, feat2, remove_mean=True):
    """
        feat1, feat2: n x d
    """
    if remove_mean:
        feat1 -= np.mean(feat1, axis=0, keepdims=1)
        feat2 -= np.mean(feat2, axis=0, keepdims=1)
    norm12 = norm(feat1.T.dot(feat2))**2
    norm11 = norm(feat1.T.dot(feat1))
    norm22 = norm(feat2.T.dot(feat2))
    return norm12 / (norm11 * norm22)


def compute_sim(feat_files):
    N = len(feat_files)
    sim = np.eye(N)
    for i in range(N):
        feat_i = np.load(feat_files[i])
        for j in range(i+1, N):
            feat_j = np.load(feat_files[j])
            sim[i, j] = KA(feat_i, feat_j, remove_mean=True)
            sim[j, i] = sim[i, j]
            print(i, j, sim[i, j], flush=True)
    return sim


if __name__ == '__main__':
    args = parse_args()
    sim = compute_sim(args.feat_files)
    np.save(args.save+'.npy', sim)
    with open(args.save+'.txt', 'w') as f:
        f.write('\n'.join(args.feat_files))
