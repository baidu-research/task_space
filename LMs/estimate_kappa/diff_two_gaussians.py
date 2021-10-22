"""
    Compute the KL divergence between two zero-mean multivariate Gaussian
    KL(S0||S1), where S0 and S1 are the covariances

    Or the cosine similarity between these two covariances
"""
import argparse
import numpy as np
from numpy.linalg import norm, inv, slogdet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('s0', type=str, help='path to a similarity matrix')
    parser.add_argument('-s1', type=str, default='',
                        help='path to another similarity matrix. If not given, '
                        'will be identity matrix')
    parser.add_argument('--metric', choices=['kl', 'corr'], default='kl')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    S0 = np.load(args.s0)
    N = S0.shape[0]
    if args.s1:
        S1 = np.load(args.s1)
        assert S0.shape == S1.shape
    else:
        S1 = np.eye(N)

    if args.metric == 'corr':
        corr = np.sum(S0 * S1) / (norm(S0) * norm(S1))
        print(corr, flush=True)

    elif args.metric == 'kl':
        sign0, logdet0 = slogdet(S0)
        assert sign0 == 1
        sign1, logdet1 = slogdet(S1)
        assert sign1 == 1

        KL = 0.5 * (np.trace(inv(S1).dot(S0)) - N + logdet1 - logdet0)
        print(KL, flush=True)
