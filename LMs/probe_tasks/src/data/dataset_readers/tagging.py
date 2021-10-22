from typing import Dict, List, Optional, Set, Union
import logging
from torch import FloatTensor

from src.contextualizers import Contextualizer
from src.data.dataset_readers import TruncatableDatasetReader

logger = logging.getLogger(__name__)


class TaggingDatasetReader(TruncatableDatasetReader):
    """
    A base DatasetReader for tagging tasks.

    Parameters
    ----------
    contextualizer: Contextualizer, optional (default=``None``)
        If provided, it is used to produce contextualized representations of the text.
    max_instances: int or float, optional (default=``None``)
        The number of instances to use during training. If int, this value is taken
        to be the absolute amount of instances to use. If float, this value indicates
        that we should use that proportion of the total training data. If ``None``,
        all instances are used.
    seed: int, optional (default=``0``)
        The random seed to use.
    """
    def _read_dataset(self,
                      file_path: str,
                      count_only: bool = False,
                      keep_idx: Optional[Set[int]] = None,
                      learn_vocab: bool = False):
        """
        Yield instances from the file_path.

        Parameters
        ----------
        file_path: str, required
            The path to the data file.
        count_only: bool, optional (default=``False``)
            If True, no instances are returned and instead a dummy object is
            returned. This is useful for quickly counting the number of instances
            in the data file, since creating instances is relatively expensive.
        keep_idx: Set[int], optional (default=``None``)
            If not None, only yield instances whose index is in this set.
        learn_vocab: return only labels, for learning a vocabulary. Bool (default False)
        """
        raise NotImplementedError


    def _learn_vocab(self, file_path: str):
        vocab = dict()
        def update_dict(vocab, keys):
            for key in keys:
                if key not in vocab:
                    vocab[key] = len(vocab)
        for labels in self._read_dataset(file_path=file_path, learn_vocab=True):
            update_dict(vocab, labels)
        return vocab
