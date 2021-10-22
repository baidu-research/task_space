from typing import Optional, Set
import logging
import random

from src.data.dataset_readers import TaggingDatasetReader

logger = logging.getLogger(__name__)

FIELDS = ["form", "tag"]


def parse_sentence(sentence: str):
    annotated_sentence = []

    lines = [line for line in sentence.split("\n") if line]

    for line_idx, line in enumerate(lines):
        annotated_token = dict(zip(FIELDS, line.split("\t")))
        annotated_sentence.append(annotated_token)
    return annotated_sentence


def lazy_parse(text: str):
    """
        Cache parsed data file
    """
    parsed = []
    for sentence in text.split("\n\n"):
        if sentence:
            parsed.append(parse_sentence(sentence))
    return parsed


class GrammaticalErrorCorrectionDatasetReader(TaggingDatasetReader):
    """
    Reads a file in a format where each line is:

    word<tab>tag

    and sentences are separated by newlines.

    This DatasetReader is used to process the FCE-public grammatical error
    correction data.

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
        learn_vocab: return only labels, for learning a vocabulary
        shuffle_rng: random seed for shuffling the order of instances
        """
        if count_only:
            logger.info("Counting grammatical error correction instances in dataset at: %s", file_path)
        else:
            logger.info("Reading grammatical error correction data from dataset at: %s", file_path)

        with open(file_path) as tagging_file:
            parsed = lazy_parse(tagging_file.read())
            ids = [_ for _ in range(len(parsed))]
            if shuffle_rng is not None:
                random.Random(shuffle_rng).shuffle(ids)
            #for i, annotation in enumerate(lazy_parse(tagging_file.read())):
            for i in ids:
                annotation = parsed[i]
                if count_only:
                    yield 1
                    continue
                if keep_idx is not None and i not in keep_idx:
                    continue
                tokens = [x["form"] for x in annotation]
                if learn_vocab:
                    yield [x["tag"] for x in annotation]
                else:
                    yield self.text_to_instance(tokens,
                                                [x["tag"] for x in annotation])
