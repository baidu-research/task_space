"""
   Extract contextualized word representations for the probe tasks
   Similar to ../estimate_kappa/extract_feature.py, but now saved as .hdf5 file 
"""

import argparse
import h5py
import json
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ckpt import load_ckpt
from data import one_pass_data_batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_name', type=str,
                        help='name of the huggingface ckpt. See ../ckpts.txt')
    parser.add_argument('data', type=str,
                        help='path to a text data file')
    parser.add_argument('--subword-to-word', choices=['first', 'last', 'average'],
                        default='last',
                        help='which subword embedding to use as the word embedding, '
                        'default to the last')
    parser.add_argument('-b', type=int, default=2,
                        help='batch size (default: 2)')
    parser.add_argument('-s', type=str,
                        help='path to save the extracted features, .hdf5 file')
    return parser.parse_args()


def extract_for_batch(ckpt, batch, word_stf, word_edt, bpe_tok):
    Outputs = ckpt(**batch)
    feat = Outputs.last_hidden_state.cpu().data.numpy()
    if args.subword_to_word == 'first':
        word_feat = [feat[i, stf] for i, stf in enumerate(word_stf)]
    elif args.subword_to_word == 'last':
        word_feat = [feat[i, edt-1] for i, edt in enumerate(word_edt)]
    elif args.subword_to_word == 'average':
        word_feat = []
        for i in range(len(word_stf)):
            n_words = len(word_stf[i])
            word_feat_i = [
                np.mean(
                    feat[i, word_stf[i][j]: word_edt[i][j]], axis=0)
                for j in range(n_words)
            ]
            word_feat.append(word_feat_i)
    return word_feat  # batch x #words x dim 


if __name__ == '__main__':
    args = parse_args()
    ckpt, tokenizer, bpe_tok = load_ckpt(args.ckpt_name)
    total_sentences = 0
    sentence_to_index = {}
    data_producer = one_pass_data_batches(args.data, tokenizer, args.b,
                                          bpe_tok, return_sentence=True)
    with h5py.File(args.s, 'w') as fout:
        for batch, w_stf, w_edt, sentences in data_producer:
            embed = extract_for_batch(ckpt, batch, w_stf, w_edt, bpe_tok)
            for i, sentence in enumerate(sentences):
                key = str(total_sentences + i)
                sentence_to_index[sentence] = key
                fout.create_dataset(
                        str(key),
                        embed[i].shape, dtype='float32',
                        data=embed[i]
                )
            total_sentences += len(sentences)
            if total_sentences % 64 == 0:
                print("Done {} sentences".format(total_sentences), flush=True)

        # add sentence index after done
        sentence_index_dataset = fout.create_dataset(
                "sentence_to_index",
                (1,),
                dtype=h5py.special_dtype(vlen=str))
        sentence_index_dataset[0] = json.dumps(sentence_to_index)
        
    if total_sentences % 64 != 0:
        print("Done {} sentences".format(total_sentences), flush=True)
