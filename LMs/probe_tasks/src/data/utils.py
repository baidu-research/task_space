from ..contextualizers import Contextualizer


def merge_vocab(vocabs):
    all_vocab = dict()
    for vocab in vocabs:
        for k in vocab:
            if k not in all_vocab:
                all_vocab[k] = len(all_vocab)
    return all_vocab


def learn_vocab(task):
    dummy_reader = task['data_reader'](Contextualizer(), **task.get('kwargs', {}))
    vocab_train = dummy_reader._learn_vocab(task['train'])
    vocab_dev = dummy_reader._learn_vocab(task['dev'])
    vocab = merge_vocab([vocab_train, vocab_dev])
    return vocab
