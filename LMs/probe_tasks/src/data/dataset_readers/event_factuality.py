from typing import Optional, Set
import json
import logging
import random

from src.data.dataset_readers import TruncatableDatasetReader

logger = logging.getLogger(__name__)

# There are three additional columns after pos: feats, head, and the deprel.
# We write the data reader so that it still works if it's missing
# those two columns (since they're non-essential anyway).

FIELDS = ['tokens', 'indices', 'labels']

def lazy_parse(file_path: str):
    """
        Parse the entire data file and store in cache
    """
    parsed = []
    with open(file_path) as event_factuality_data_file:
        event_factuality_data = json.load(event_factuality_data_file)
    for sentence_id, sentence_data in event_factuality_data.items():
        tokens = sentence_data["sentence"]
        predicate_indices = sentence_data["predicate_indices"]
        # indices start from 0
        predicate_indices = [i-1 for i in predicate_indices]
        labels = sentence_data["labels"]
        if not predicate_indices or not labels:
            continue
        parsed.append(dict(zip(FIELDS, (tokens, predicate_indices, labels))))
    return parsed


class EventFactualityDatasetReader(TruncatableDatasetReader):
    """
    A dataset reader for the processed ItHappened event factuality dataset.

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
    lazy : ``bool``, optional (default=``False``)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    """
    def _read_dataset(self,
                      file_path: str,
                      count_only: bool = False,
                      keep_idx: Optional[Set[int]] = None,
                      shuffle_rng=None):
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
        shuffle_rng: random seed for shuffling order of instances, suggested
        """
        if count_only:
            logger.info("Counting instances from ItHappened file at: %s", file_path)
        else:
            logger.info("Reading instances from ItHappened file at: %s", file_path)
        parsed = lazy_parse(file_path)
        ids = [_ for _ in range(len(parsed))]
        if shuffle_rng is not None:
            random.Random(shuffle_rng).shuffle(ids)
        for i in ids:
            sample = parsed[i]
            if count_only:
                yield 1
                continue
            if keep_idx is not None and i not in keep_idx:
                continue
            yield self.text_to_instance(
                    sample['tokens'],
                    sample['labels'],
                    sample['indices']
                  )
