"""
    Refer to
    https://github.com/nelson-liu/contextual-repr-analysis/blob/master/contexteval/data/dataset_readers/truncatable_dataset_reader.py
"""
from typing import List, Optional, Set, Union
import logging
import random
from src.contextualizers import Contextualizer

logger = logging.getLogger(__name__)


class TruncatableDatasetReader:
    """
    A base DatasetReader with the ability to return only a subset of the generated
    instances.

    Parameters
    ----------
    max_instances: int or float, optional (default=``None``)
        If None, use the entire dataset.
        Otherwise, it is the number of instances to use during training.
        If int, this value is taken to be the absolute amount of instances
        to use. If float, this value indicates that we should use that 
        proportion of the total training data.
    seed: int, optional (default=``0``)
        The random seed to use for sampling a subset of data
    """
    def __init__(self,
                 contextualizer: Contextualizer,
                 max_instances: Optional[Union[int, float]] = None,
                 seed: int = 0, # used to sample a subset of the data
                 ) -> None:
        self._contextualizer = contextualizer
        self._max_instances = max_instances
        self._keep_idx: Set[int] = set()
        self._seed = seed
        self._rng = random.Random(seed)


    def _read(self, file_path: str, shuffle_rng=None):
        if self._max_instances is None or self._max_instances == 1.0:
            yield from self._read_dataset(file_path, shuffle_rng=shuffle_rng)
        else: # We want to truncate the dataset.
            if not self._keep_idx:
                total_num_instances = self._count_instances_in_dataset(file_path)
                # Generate the indices to keep
                dataset_indices = list(range(total_num_instances))
                self._rng.shuffle(dataset_indices)
                if isinstance(self._max_instances, int):
                    num_instances_to_keep = self._max_instances
                else:
                    num_instances_to_keep = int(
                        self._max_instances * total_num_instances)

                if num_instances_to_keep > total_num_instances:
                    logger.warning("Raw number of instances to keep is %s, but total "
                                   "number of instances in dataset is %s. Keeping "
                                   "all instances...", num_instances_to_keep, total_num_instances)
                self._keep_idx.update(dataset_indices[:num_instances_to_keep])
                logger.info("Keeping %s instances", len(self._keep_idx))

            # We know which instances we want to keep, so yield from the reader,
            # taking only those instances.
            yield from self._read_dataset(file_path=file_path,
                                          keep_idx=self._keep_idx,
                                          shuffle_rng=shuffle_rng)

    def _read_dataset(self,
                      file_path: str,
                      count_only: bool = False,
                      keep_idx: Optional[Set[int]] = None):
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
        """
        raise NotImplementedError

    def _count_instances_in_dataset(self, file_path: str):
        num_instances = 0
        for instance in self._read_dataset(file_path=file_path, count_only=True):
            num_instances += 1
        return num_instances

    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         labels: List[str] = None,
                         label_indices: List[int] = None):
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        labels : ``List[str]``, optional, (default = None).
            The labels for the words in the sentence.

        Returns
        -------
        A dictionary:
            'tokens': List[str], a list of words
            'representations': torch.Tensor, seq_len x dim
            labels : ``SequenceLabelField``
                The labels (only if supplied)
        """
        instance = dict()
        instance['tokens'] = tokens
        instance['labels'] = labels
        instance['label_indices'] = label_indices
        instance['representations'] = self._contextualizer([tokens])[0]
        return instance


