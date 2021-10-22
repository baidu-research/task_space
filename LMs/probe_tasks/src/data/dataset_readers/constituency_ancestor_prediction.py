from typing import List, Optional, Set, Union
import logging
import random

from nltk.tree import ParentedTree

from src.contextualizers import Contextualizer
from src.data.dataset_readers import TaggingDatasetReader

logger = logging.getLogger(__name__)

PTB_SPECIAL_TOKENS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]"
}


class ConstituencyAncestorPredictionDatasetReader(TaggingDatasetReader):
    """
    Reads a file with linearized trees in Penn Treebank Format and produces instances
    suitable for use by an auxiliary classifier that aims to predict, given a word representation,
    the constituency label of an ancestor (parent/grandparent/great-grandparent).

    Parameters
    ----------
    ancestor: str, optional (default=``"parent"``)
        The tree position to take as the label. One of "parent", "grandparent", or "greatgrandparent",
        where "parent" indicates that we predict the constituency label of the word's parent,
        "grandparent" indicates that we predict the constituency label of the word's grandparent,
        and "greatgrandparent" indicates that we predict the constituency label of the word's
        great-grandparent.
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
    def __init__(self,
                 contextualizer: Contextualizer,
                 max_instances: Optional[Union[int, float]] = None,
                 seed: int = 0,
                 ancestor: str = "parent"
                ) -> None:
        super().__init__(
            contextualizer=contextualizer,
            max_instances=max_instances,
            seed=seed)
        if ancestor not in ["parent", "grandparent", "greatgrandparent"]:
            raise ConfigurationError(
                "Got invalid value {} for ancestor. "
                "ancestor must be one of \"parent\", \"grandparent\", \"greatgrandparent\".")
        self._ancestor = ancestor

        self.cache = []
        self.cached = False

    def _read_dataset(self,
                      file_path: str,
                      count_only: bool = False,
                      keep_idx: Optional[Set[int]] = None,
                      learn_vocab: bool = False,
                      shuffle_rng = None):
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
        shuffle_rng: random seed to shuffle the instance orders
        """
        if count_only:
            logger.info("Counting instances (%s) in file at: %s",
                        self._ancestor, file_path)
        else:
            logger.info("Reading instances (%s) from lines in file at: %s",
                        self._ancestor, file_path)

        # one pass through the data file
        if count_only or learn_vocab or not self.cache:
            with open(file_path) as input_file:
                for line in input_file:
                    clean_line = line.rstrip("\n")
                    if line.startswith("#"):
                        continue
                    if count_only:
                        yield 1
                        continue
                    # Create a ParentedTree from the line
                    tree = ParentedTree.fromstring(clean_line)
                    # Remove the empty top layer
                    if tree.label() == "VROOT" or tree.label() == "TOP" or tree.label() == "":
                        tree = tree[0]
                        # Hacky way of deleting the parent
                        tree._parent = None
                    # Get the associated tokens and tags, depending on the ancestor
                    tokens, labels = self.get_example(tree, self._ancestor)
                    # Filter the tokens for special PTB tokens
                    tokens = [PTB_SPECIAL_TOKENS.get(token, token) for token in tokens]

                    if learn_vocab:
                        yield labels
                    else:
                        self.cache.append((tokens, labels))
            self.cached = True
        
        # yield from cache, with shuffling if required
        if not count_only and not learn_vocab and self.cached:
            ids = [_ for _ in range(len(self.cache))]
            if shuffle_rng is not None:
                random.Random(shuffle_rng).shuffle(ids)
            for i in ids:
                if keep_idx is not None and i not in keep_idx:
                    continue
                tokens, labels = self.cache[i]
                yield self.text_to_instance(tokens=tokens,
                                            labels=labels)

    def get_example(self,  # type: ignore
                    tree: ParentedTree,
                    ancestor: str):
        """
        Given a ParentedTree, extract the labels of the parents,
        grandparents, or greatgrandparents.

        Parameters
        ----------
        tree: ParentedTree
            ParentedTree to extract the example from.
        ancestor: str
            Whether the labels should be the parent, grandparent, or great-grandparent
            of each leaf.
        """
        tokens = tree.leaves()
        labels: List[str] = []
        for child in tree:
            if isinstance(child, ParentedTree):
                if len(list(child.subtrees())) > 1:
                    labels.extend(self.get_example(child, self._ancestor)[1])
                else:
                    labels.append(self._get_label(child, self._ancestor))
        return tokens, labels

    def _get_label(self,  # type: ignore
                   tree: ParentedTree,
                   ancestor: str):
        levels_up = {"parent": 1, "grandparent": 2, "greatgrandparent": 3}
        label = tree
        for i in range(levels_up[ancestor]):
            label = label.parent()
            if label is None:
                return 'None'
        return label.label()
