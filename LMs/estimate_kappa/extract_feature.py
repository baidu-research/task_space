"""
    Refer to https://github.com/ethanjperez/pytorch-pretrained-BERT/blob/master/examples/extract_features.py
    A word's embedding is its last (default) subword's embedding
    Refer to https://github.com/huggingface/transformers/issues/64 and 
    https://www.mrklie.com/post/2020-09-26-pretokenized-bert/
"""

import argparse
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ckpt import load_ckpt
from data import one_pass_data_batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_name', type=str,
                        help='name of the checkpoint, refer to ../ckpts.txt')
    parser.add_argument('data', type=str,
                        help='path to a text data file')
    parser.add_argument('--subword-to-word', choices=['first', 'last', 'average'],
                        default='last',
                        help='which subword embedding to use as the word embedding, '
                        'default to the last')
    parser.add_argument('-b', type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument('-s', type=str,
                        help='path to save the extracted features, .npz file')
    parser.add_argument('--save-at', type=int, nargs='+',
                        help='save at these many sentences '
                        '(default to only save once at finish)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ckpt, tokenizer, bpe_tok = load_ckpt(args.ckpt_name)
    embds = []
    word_offset = [] # 2-col, word's row id and col id
    total_sentences = 0
    for Inputs, word_starts_from, word_ends_at in \
            one_pass_data_batches(args.data, tokenizer, args.b, bpe_tok):
        Outputs = ckpt(**Inputs)
        feat = Outputs.last_hidden_state.cpu().data.numpy()
        if args.subword_to_word == 'first':
            word_feat = [feat[i, stf] for i, stf in enumerate(word_starts_from)]
        elif args.subword_to_word == 'last':
            word_feat = [feat[i, edt-1] for i, edt in enumerate(word_ends_at)]
        elif args.subword_to_word == 'average':
            word_feat = []
            for i in range(len(word_starts_from)):
                n_words = len(word_starts_from[i])
                word_feat_i = [
                    np.mean(
                        feat[i, word_starts_from[i][j]: word_ends_at[i][j]], axis=0)
                    for j in range(n_words)
                ]
                word_feat.append(word_feat_i)
        embds.append(np.vstack(word_feat))

        for i, this_line in enumerate(word_starts_from):
            for j in range(len(this_line)):
                word_offset.append([total_sentences+i, j])

        #assert embds[-1].shape[0] == word_offset[-1].shape[0]

        total_sentences += len(word_starts_from)
        print("Done {} sentences".format(total_sentences), flush=True)

        if args.save_at is not None and total_sentences in args.save_at:
            np.savez(
                args.s+'.'+str(total_sentences),
                x=np.vstack(embds),
                offset=np.vstack(word_offset)
            )
            embds = []
            word_offset = []
            print("Saved {} sentences".format(total_sentences), flush=True)

    # handle the not saved tail, or if only save once at end
    if len(word_offset) > 0:
        np.savez(
            args.s + '.' + str(total_sentences),
            x=np.vstack(embds),
            offset=np.vstack(word_offset)
        )
        print("Saved {} sentences".format(total_sentences), flush=True)
