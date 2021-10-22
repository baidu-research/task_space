"""
    Refer to https://github.com/nelson-liu/contextual-repr-analysis/blob/master/contexteval/contextualizers/precomputed_contextualizer.py
"""
import json
import logging
from typing import List, Optional

import h5py
import torch

from src.contextualizers import Contextualizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PrecomputedContextualizer(Contextualizer):
    """
    This "contextualizer" simply reads representations from one or multiple
    hdf5 files provided upon construction.
    
    Each hdf5 has the following two groups of data:
        sentence_to_index: dict that maps each sentence (string) to a index
        indexed representaion: representation (words x dim) matrix for each
            indexed sentence
        
    If multiple hdf5 files, they must be built based on the same set of
    sentencces. The returned are concatenated features.

    Parameters
    ----------
    representations_paths: list of paths
        A list of Paths to multiple HDF5 files with the representations.
    """
    def __init__(self,
                 representations_paths: List[str],
                 ) -> None:
        super(PrecomputedContextualizer, self).__init__()
        # if `file_path` is a URL, redirect to the cache
        self.paths = representations_paths
        # Read the HDF5 file.
        self._representations = [h5py.File(_, 'r') for _ in self.paths]
        # Get the sentences to index mapping
        self._sentence_to_index = json.loads(
                self._representations[0].get('sentence_to_index')[0])
        # some sanity checks
        for i in range(1, len(self._representations)):
            this_sentence_to_index = json.loads(
                    self._representations[i].get('sentence_to_index')[0])
            assert this_sentence_to_index == self._sentence_to_index
        self.dim = self.get_dim()

    def forward(self, sentences: List[List[str]]) -> torch.FloatTensor:
        """
        Parameters
        ----------
        sentences: List[List[str]]
            A batch of sentences. len(sentences) is the batch size, and each sentence
            itself is a list of strings (the constituent words). If the batch is padded,
            the expected padding token in the Python ``None``.

        Returns
        -------
        representations: List[FloatTensor]
            A list with the contextualized representations of all words in an input sentence.
            Each inner FloatTensor is of shape (seq_len, repr_dim), and an outer List
            is used to store the representations for each input sentence.
        """
        batch_representations = []

        for sentence in sentences:
            this_sentence = " ".join([x for x in sentence if x is not None])
            this_id = self._sentence_to_index[this_sentence]
            # read multiple representations (words x dim) into a list
            representation = [torch.FloatTensor(reptn[this_id]) 
                    for reptn in self._representations]
            representation = torch.hstack(representation)
            batch_representations.append(representation)
        return batch_representations

    def get_dim(self):
        """
            Get dimension of features
        """
        dims = [rptn['1'].shape[1] for rptn in self._representations]
        return sum(dims)

