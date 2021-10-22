"""
    Compute kappa matrix for a list of checkpoints from huggingface.
    Each checkpoint has its feature stored as /path/to/ckpt_name.sentences.npz
"""
import argparse
import glob
import numpy as np
import os
import pylab
import sys
from numpy.linalg import norm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('feat_root', type=str,
                        help='path to feature dir')
    parser.add_argument('ckpt_name_file', type=str,
                        help='a txt file, where each row a huggingface ckpt name')
    parser.add_argument('save', type=str, help='path to save kappa matrix')
    parser.add_argument('--max-sentence', type=int,
                        help='only use the first `max_sentence` '
                        'sentences. If not given, will use all that can be loaded')
    parser.add_argument('--no-mean-removal', action='store_true',
                        help='no mean removal before computing kappa, not suggested')
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


def compute_sim(feat1, feat2):
    """
        Compute similarity between two models' extracted features
    """
    # word to word align of the features
    offset1 = feat1['offset']
    offset2 = feat2['offset']
    dtype={
           'names': ['f{}'.format(i) for i in range(2)],
           'formats': 2*[offset1.dtype]
          }
    _, comm1, comm2 = np.intersect1d(
                            offset1.view(dtype),
                            offset2.view(dtype),
                            return_indices=True
                      )
    sim = KA(feat1['x'][comm1], feat2['x'][comm2],
             remove_mean=not args.no_mean_removal)
    return sim


def load_merge_segments(list_of_npz, max_sentence=None):
    """
        Given a list of /path/to/ckptName.sentences.npz 's. Merge them
        and return a dictionary with
        'x': feature matrix, sample x dimension
        'offset': 2d matrix, each row is [line number, word offset within line] 
    """
    loaded={'x': [], 'offset': []}
    for npz_file in list_of_npz:
        npz = np.load(npz_file)
        x = npz['x']
        offset = npz['offset']
        if max_sentence:
            x = x[offset[:, 0]<max_sentence]
            offset = offset[offset[:, 0]<max_sentence]
        loaded['x'].append(x)
        loaded['offset'].append(offset)
    loaded['x'] = np.vstack(loaded['x'])
    loaded['offset'] = np.vstack(loaded['offset'])
    return loaded


def get_npzs(feat_root, ckpt_name, max_sentence=None):
    all_files = glob.glob(os.path.join(feat_root, ckpt_name+'.*.npz'))
    if max_sentence:
        segs = [int(f.split('/')[-1].split('.')[1]) for f in all_files]
        # a bit hacky, e.g., seg=[10, 20, 30, 40], and max_sentence=25
        # we need to load the 1st, 2nd and 3rd segment
        segs.sort()
        inds = np.where(np.asarray(segs)<=max_sentence)[0]
        if len(inds):
            inds = inds[-1]
            if inds < len(segs)-1:
                inds += 1
        else:
            inds = 0
        files = [os.path.join(feat_root,
                              '{}.{}.npz'.format(ckpt_name, segs[i]))
                 for i in range(inds+1)]
        return files
    else:
        return all_files


if __name__ == '__main__':
    args = parse_args()
    all_ckpts = open(args.ckpt_name_file, 'r').read().strip().split()
    N = len(all_ckpts)
    kappa = np.eye(N)
    
    for i in range(N):
        feats_i = get_npzs(args.feat_root, all_ckpts[i], args.max_sentence)
        feats_i_loaded = load_merge_segments(feats_i, args.max_sentence)
        for j in range(i+1, N):
            feats_j = get_npzs(args.feat_root, all_ckpts[j], args.max_sentence)
            feats_j_loaded = load_merge_segments(feats_j, args.max_sentence)
            kappa[i, j] = compute_sim(feats_i_loaded, feats_j_loaded)
            kappa[j, i] = kappa[i, j]
            print("{} {} {}".format(
                    all_ckpts[i], all_ckpts[j], kappa[i, j]), flush=True)
    
    np.save(args.save, kappa)
