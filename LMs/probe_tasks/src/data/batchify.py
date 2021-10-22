"""
    Since we only use softmax classifier for specified tokens,
    we can ignore representations for tokens that do not have a label

    Right now, only work for
    1. tagger (each token in a sentence has a categorical label)
    2. selective tagger (tokens at selected positions have categorical labels)
    3. selective regressor (tokens at selected positions have a continuous
        target value to be mapped to)
"""
import torch
import numpy as np
from functools import reduce
from torch.nn.utils.rnn import pad_sequence
np.random.seed(0)


def yield_batch(list_of_inputs, list_of_labels):
    """
        Group list of features and labels into batch
        list_of_inputs: list of Length_i x dim tensor
        list_of_labels: list [Length_1,...]
        The two lists' lengths are both `batch_size`
    """
    inputs = pad_sequence(list_of_inputs, batch_first=True)
    targets = pad_sequence(list_of_labels, batch_first=True, padding_value=-1)
    mask = pad_sequence(
                [torch.ones_like(label, dtype=bool) for label in list_of_labels],
                batch_first=True
           ) 
    return inputs, targets, mask


class data_producer:
    """
        Go through the data file and extracted features to prepare input and
        target for softmax classifier, grouping samples into specified batch
        size.
    """
    def __init__(self, reader, file_path,
                 task_type='tagger', vocab=None):
        """
            reader: A data reader constructed using extracted features
            file_path: data file
            batch_size: int, default to 128
            task_type: ['tagger', 'selective_tagger', 'selective_regressor'].
                default to 'tagger'
            vocab: vocabulary of output space, mapping each label to a integer.
                If not given, will contruct from training data.
        """
        self.reader = reader
        self.file_path = file_path
        self.task_type = task_type
        
        # vocab for output space. Map tags to integer index
        self.vocab = vocab
        if 'tagger' in task_type and self.vocab is None:
            self.vocab = reader._learn_vocab(file_path) 

    def batches(self, batch_size, shuffle_rng=None):
        """
            Create batch of data, where each batch includes `batch_size` of sequences
            shuffle_rng: random seed for shuffling the sequences if not None.
                suggested for training.
        """
        batch_inputs = []
        batch_targets = []

        for instance in self.reader._read(self.file_path, shuffle_rng=shuffle_rng):
            if self.task_type == 'tagger':
                batch_inputs.append(instance['representations'])
                batch_targets.append(
                    torch.tensor([self.vocab[_] for _ in instance['labels']],
                                 dtype=torch.long if len(self.vocab)>1 else torch.float)
                )
            elif self.task_type == 'selective_tagger':
                batch_inputs.append(
                    instance['representations'][instance['label_indices']])
                batch_targets.append(
                    torch.tensor([self.vocab[_] for _ in instance['labels']],
                                 dtype=torch.long if len(self.vocab)>1 else torch.float)
                )
            elif self.task_type == 'selective_regressor':
                batch_inputs.append(
                    instance['representations'][instance['label_indices']])
                batch_targets.append(
                    torch.tensor(instance['labels'], dtype=torch.float)
                )
            else:
                raise ValueError('Unknown task type {}'.format(self.task_type))

            if len(batch_inputs) == batch_size:
                yield yield_batch(batch_inputs, batch_targets)
                batch_inputs = []
                batch_targets = []
        if len(batch_inputs):  # the last few instances
                yield yield_batch(batch_inputs, batch_targets)
