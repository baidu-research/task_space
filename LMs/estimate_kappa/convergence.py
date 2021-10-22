import argparse
import glob
import numpy as np
import pylab
import subprocess
from numpy.linalg import norm, inv, slogdet 

pylab.rcParams['font.size'] = 20

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+',
                        choices=['wikitext2', '1b-10K'])
    parser.add_argument('--metric', choices=['kl', 'corr'], default='kl')
    parser.add_argument('--iso', action='store_true',
                        help='isotropic Gaussian baseline')
    parser.add_argument('-s', type=str, help='name to save')
    return parser.parse_args()



def read_and_compute(data, metric, iso):
    if data == 'wikitext2':
        kappas = glob.glob('kappa.wiki2.[0-9]*.npy')
        kappa_star = 'kappa.wiki2.all.npy'
        data_file = '../text/wiki.train.len_noLess_10.tokens'
    elif data == '1b-10K':
        kappas = glob.glob('kappa.1b-10K.[0-9]*.npy')
        kappa_star = 'kappa.1b-10K.all.npy'
        data_file = '../text/billion_10K.txt'

    sentences = []
    for kappa in kappas:
        sents = kappa.split('/')[-1].split('.')[2] 
        assert sents.isnumeric()
        sentences.append(int(sents))
    ids = np.argsort(sentences)
    sentences = [sentences[_] for _ in ids]
    kappas = [kappas[_] for _ in ids]
    kappas.append(kappa_star)
    
    word_counts = []
    with open(data_file, 'r') as f:
        line_cnt = 0
        wc = 0
        for line in f:
            line_cnt += 1
            wc += len(line.strip().split())
            if line_cnt in sentences:
                word_counts.append(wc)
    word_counts.append(wc)
    diffs = []
    if iso:
        word_counts.insert(0, 1)
        diffs.append(float(
                    subprocess.check_output(['python', 'diff_two_gaussians.py',
                                             kappas[-1],
                                             '--metric', args.metric])
                   )) 
    for kappa in kappas:
        diffs.append(float(
                     subprocess.check_output(['python', 'diff_two_gaussians.py',
                                             kappas[-1], '-s1', kappa,
                                             '--metric', args.metric])
                    ))
    return word_counts, diffs
    

if __name__ == '__main__':
    args = parse_args()
    for data in args.data:
        word_counts, diffs = read_and_compute(data, args.metric, args.iso)
        if data == 'wikitext2':
            label = 'wikitext2-train'
        elif data == '1b-10K':
            label = '1B-10K'
        pylab.semilogx(word_counts, diffs, '.-', label=label)
    pylab.xlabel('word counts in probing data')
    if args.metric == 'kl':
        pylab.ylabel(r'$KL(\kappa^*||\kappa)$')
    elif args.metric == 'corr':
        pylab.ylabel(r'$cos(vec(\kappa^*), vec(\kappa))$')
    pylab.legend()
    pylab.tight_layout()
    if args.s:
        pylab.savefig('{}'.format(args.s))
    else:
        pylab.show()


