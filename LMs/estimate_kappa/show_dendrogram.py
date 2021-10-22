"""
    Refer to https://stackoverflow.com/questions/48539558/swap-leafs-of-python-scipys-dendrogram-linkage
"""
import argparse
import pylab
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
pylab.rcParams['font.size'] = 15


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kappa', type=str, help='path to kappa matrix')
    parser.add_argument('ckpt_name_file', type=str,
                        help='path to a txt file, where i-th row is the name '
                        'of the i-th checkpoint')
    parser.add_argument('-s', type=str, help='save plot')
    args = parser.parse_args()

    
    S = np.load(args.kappa)  # similarity matrix
    D = 1.0 - S  # distance matrix
    labels = open(args.ckpt_name_file, 'r').read().strip().split('\n')
    for i, l in enumerate(labels):
        if len(labels[i].split('-')) >= 4:
            labels[i] = '-'.join(labels[i].split('-')[:3])
        if 'xlnet' in labels[i]:
            labels[i] = '-'.join(labels[i].split('-')[:2])
    N = len(labels)
    assert S.shape[0] == N

    fig, axes = pylab.subplots(1, 2)
    pylab.subplots_adjust(left=0.05, right=0.95, bottom=0.34, top=0.99, wspace=0.35)
    mng = pylab.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    # Dendrogram
    L = linkage(squareform(D), 'ward', optimal_ordering=True)
    den = dendrogram(L, ax=axes[1], color_threshold=0.9,
                     orientation='right', labels=labels,
                     leaf_font_size=12)
    # get leaves' colors
    #see https://stackoverflow.com/questions/61959602/retrieve-leave-colors-from-scipy-dendrogram
    colors = ['none'] * N
    for xs, c in zip(den['icoord'], den['color_list']):
        for xi in xs:
            if xi % 10 == 5:
                colors[(int(xi)-5) // 10] = c
    for ytick, color in zip(axes[1].get_yticklabels(), colors):
        ytick.set_color(color)

    # kappa matrix itself
    # reorder all entities, note that imshow places (0, 0) on top left corner
    S_reorder = S[den['leaves'][::-1]][:, den['leaves'][::-1]]
    im = axes[0].imshow(S_reorder, vmin=0, vmax=1)
    pylab.colorbar(im, ax=axes[0], cax=fig.add_axes([0.01, 0.34, 0.02, 0.65]))
    axes[0].get_yaxis().set_ticks([])
    axes[0].set_xticks(range(N))
    axes[0].set_xticklabels(den['ivl'][::-1], rotation=90)
    for xtick, color in zip(axes[0].get_xticklabels(), colors[::-1]):
        xtick.set_color(color)
    if args.s:
        pylab.savefig(args.s)
    else:
        pylab.show()
