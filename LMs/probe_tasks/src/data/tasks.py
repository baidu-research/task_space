import copy
import os
from allennlp.data import Vocabulary
from .dataset_readers import *
from ..training import metric

contextualizer_dir = 'contextualizers'
default_metric = {
    'acc_k=1': metric('acc', top_k=1),
    'acc_k=3': metric('acc', top_k=3)
}

pos_ptb = {'feature': 'ptb_pos.hdf5',
           'train': 'data/pos/wsj.train.conllx',
           'dev': 'data/pos/wsj.dev.conllx',
           'test': 'data/pos/wsj.test.conllx',
           'data_reader': ConllXPOSDatasetReader,
           'metric': default_metric,
           'type': 'tagger'}

st = {'feature': 'semantic_tagging.hdf5',
      'train': 'data/semantic_tagging/semtag_train.conll',
      'dev': 'data/semantic_tagging/semtag_dev.conll',
      'test': 'data/semantic_tagging/semtag_test.conll',
      'data_reader': SemanticTaggingDatasetReader,
      'metric': default_metric,
      'type': 'tagger'}

ged = {'feature': 'grammatical_error_correction.hdf5',
       'train': 'data/grammatical_error_correction/fce-public.train',
       'dev': 'data/grammatical_error_correction/fce-public.dev',
       'test': 'data/grammatical_error_correction/fce-public.test',
       'data_reader': GrammaticalErrorCorrectionDatasetReader,
       'vocab': {'c': 0, 'i': 1},  # c: correct, i: grammar error
       'metric': {'f1': metric('f1_binary', positive_label=1)},
       'type': 'tagger'}

syn_p = {'feature': 'syntactic_constituency.hdf5',
         'train': 'data/syntactic_constituency/wsj.train.trees',
         'dev': 'data/syntactic_constituency/wsj.dev.trees',
         'test': 'data/syntactic_constituency/wsj.test.trees',
         'data_reader': ConstituencyAncestorPredictionDatasetReader,
         'kwargs': {'ancestor': 'parent'},
         'metric': default_metric,
         'type': 'tagger'}

syn_gp = copy.deepcopy(syn_p)
syn_gp['kwargs'] = {'ancestor': 'grandparent'}

syn_ggp = copy.deepcopy(syn_p)
syn_ggp['kwargs'] = {'ancestor': 'greatgrandparent'}

read_tag_count = open('data/ner/tag_counts.txt', 'r').read().strip().split('\n')
ner_tag_count = dict([(row.split()[0], int(row.split()[1])) for row in read_tag_count])
ner_vocab = Vocabulary({'tags': ner_tag_count})
ner = {'feature': 'conll2003_ner.hdf5',
       'train': 'data/ner/eng.train',
       'dev': 'data/ner/eng.testa',
       'test': 'data/ner/eng.testb',
       'data_reader': NERDatasetReader,
       'vocab': ner_vocab.get_token_to_index_vocabulary('tags'),
       'metric': {'f1': metric('span_f1', vocabulary=ner_vocab, label_encoding='IOB1')},
       'type': 'tagger'}

read_tag_count = open('data/chunking/tag_counts.txt', 'r').read().strip().split('\n')
chunking_tag_count = dict([(row.split()[0], int(row.split()[1])) for row in read_tag_count])
chunking_vocab = Vocabulary({'tags': chunking_tag_count})
chunking = {'feature': 'conll2000_chunking.hdf5',
            'train': 'data/chunking/eng_chunking_train.conll',
            'dev': 'data/chunking/eng_chunking_dev.conll',
            'test': 'data/chunking/eng_chunking_test.conll',
            'data_reader': Conll2000ChunkingDatasetReader,
            'vocab': chunking_vocab.get_token_to_index_vocabulary('tags'),
            'metric': {'f1': metric('span_f1', vocabulary=chunking_vocab, label_encoding='BIO')},
            'type': 'tagger'}

ef = {'feature': 'event_factuality.hdf5',
      'train': 'data/event_factuality/it-happened_eng_ud1.2_07092017.train.json',
      'dev': 'data/event_factuality/it-happened_eng_ud1.2_07092017.dev.json',
      'test': 'data/event_factuality/it-happened_eng_ud1.2_07092017.test.json',
      'data_reader': EventFactualityDatasetReader,
      'metric': {'pearson': metric('pearson')},
      'type': 'selective_regressor'}

task_dict = {
             'pos_ptb': pos_ptb,
             'st': st,
             'ged': ged,
             'syn_p': syn_p,
             'syn_gp': syn_gp,
             'syn_ggp': syn_ggp,
             'ner': ner,
             'chunking': chunking,
             'ef': ef
            }
