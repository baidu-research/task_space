from typing import Optional, Set
import itertools
import logging
import random

from src.data.dataset_readers import TaggingDatasetReader
logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":
            return True
        else:
            return False


def lazy_parse(file_path: str):
    """
        Parse the entire data file and store in cache
    """
    parsed = []
    with open(file_path, "r") as data_file:
        for is_divider, lines in itertools.groupby(data_file, _is_divider):
            if not is_divider:
                fields = [line.strip().split() for line in lines]
                tokens, _, _, ner_tags = [list(field) for field in zip(*fields)]
                parsed.append((tokens, ner_tags))
    return parsed


class NERDatasetReader(TaggingDatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:
    WORD POS-TAG CHUNK-TAG NER-TAG
    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.
    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)

    The label encoding method is 'IOB1'.

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
                      learn_vocab: bool = False,
                      shuffle_rng=None
                      ):
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
        learn_vocab: return only the NER tag for learning a vocabulary
        shuffle_rng: random seed for shuffling order of instances, suggested
        """
        if count_only:
            logger.info("Counting POS tagging instances from CoNLL-X formatted dataset at: %s", file_path)
        else:
            logger.info("Reading POS tagging data from CoNLL-X formatted dataset at: %s", file_path)
        
        parsed = lazy_parse(file_path)
        ids = [_ for _ in range(len(parsed))]
        if shuffle_rng is not None:
            random.Random(shuffle_rng).shuffle(ids)
        for i in ids:
            annotated = parsed[i]
            if count_only:
                yield 1
                continue
            if keep_idx is not None and i not in keep_idx:
                continue
            tokens, tags = annotated
            if learn_vocab:
                yield tags
            else:
                yield self.text_to_instance(tokens, tags)
