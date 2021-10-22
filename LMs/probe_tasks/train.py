import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src import *


LMs = open('../ckpts.txt', 'r').read().strip().split()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=list(task_dict.keys()),
                        help='task name')
    parser.add_argument('-lms', type=str, nargs='+', choices=LMs,
                        help='names of language model checkpoints. '
                        'Refer to ../ckpts.txt')
    parser.add_argument('-b', type=int, default=64,
                        help='batch size (number of sentences)')
    parser.add_argument('-e', type=int, default=40,
                        help='number of epochs')
    parser.add_argument('-lr', type=float, default=1e-3,
                        help='Initial learning rate in adam optimizer')
    parser.add_argument('-f', '--fraction', type=float,
                        help='use only a fraction of training set')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='do not shuffle sentences from epoch to epoch. '
                        'Not suggested')
    parser.add_argument('-w', '--weight-decay', type=float, default=0.,
                        help='weight decay')
    parser.add_argument('--decay-at', nargs='+', type=int,
                        help='If not given, fix learning rate. Otherwise, '
                        'decay the learning rate by a factor of 0.1')
    return parser.parse_args()


def create_data_producer(task):
    """
        Create data producer for a given task
    """
    if 'vocab' in task:
        vocab = task['vocab']
        out_dim = len(vocab)
    elif 'tagger' in task['type']:  # classification task: build the vocab right now
        vocab = learn_vocab(task)
        out_dim = len(vocab)
    else: # regression task: no vocab to be built
        vocab = None
        out_dim = 1
    task_additional_kwargs = task.get('kwargs', {})

    train_reader = task['data_reader'](
                        extracted,
                        max_instances=args.fraction,
                        **task_additional_kwargs
                   )
    train_producer = data_producer(train_reader, task['train'], task['type'],
                                   vocab)

    dev_reader = task['data_reader'](extracted, **task_additional_kwargs)
    dev_producer = data_producer(dev_reader, task['dev'], task['type'],
                                 vocab)

    test_reader = task['data_reader'](extracted, **task_additional_kwargs)
    test_producer = data_producer(test_reader, task['test'], task['type'],
                                  vocab)
    return train_producer, dev_producer, test_producer, out_dim


if __name__ == '__main__':
    args = parse_args()
    print(args, flush=True)
    task = task_dict[args.task]
    extracted = PrecomputedContextualizer(
                    [os.path.join(
                        contextualizer_dir, lm, task['feature'])
                     for lm in args.lms])
    
    train_producer, dev_producer, test_producer, out_dim = create_data_producer(task)
    torch.manual_seed(0)
    probe = simple_token_head(extracted.dim, out_dim,
                              is_regression='tagger' not in task['type'])
    if use_gpu:
        probe.cuda()
    optimizer = optim.Adam(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.decay_at:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                        milestones=args.decay_at, gamma=0.1)

    print('Running evaluation before any training ...', flush=True)
    valid_metrics = eval_model(probe, dev_producer.batches(args.b), task['metric'])
    test_metrics = eval_model(probe, test_producer.batches(args.b), task['metric'])
    print_evaluation(valid_metrics, 0, 'dev')
    print_evaluation(test_metrics, 0, 'test')
    shuffle_rng = None

    total_iter = 0
    for e in range(args.e):
        if not args.no_shuffle:
            shuffle_rng = e
        total_iter = train_one_epoch(
                        probe,
                        train_producer.batches(args.b, shuffle_rng=shuffle_rng),
                        optimizer,
                        epoch_id=e, total_iter=total_iter
                     )
        valid_metrics = eval_model(
                probe, dev_producer.batches(args.b), task['metric'])
        test_metrics = eval_model(
                probe, test_producer.batches(args.b), task['metric'])
        print_evaluation(valid_metrics, e+1, 'dev')
        print_evaluation(test_metrics, e+1, 'test')
        if args.decay_at:
            scheduler.step()
