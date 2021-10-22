import torch
from allennlp.training.metrics import *


use_gpu = torch.cuda.is_available()


class metric():
    """
        A wrapper over AllenNLP's metrics
        https://github.com/allenai/allennlp/tree/main/allennlp/training/metrics 

        make sure reading a metric returns a single scalar
    """
    # map from string to allennlp metric classes
    metric_map = {
                  'acc': CategoricalAccuracy,
                  'bleu': BLEU,
                  'entropy': Entropy,
                  'f1_binary': F1Measure,
                  'f1_multiclass': FBetaMeasure,
                  'mae': MeanAbsoluteError,
                  'pearson': PearsonCorrelation,
                  'ppl': Perplexity,
                  'span_f1': SpanBasedF1Measure
                 }
    return_tag = {
                  'f1_binary': 'f1',
                  'f1_multiclass': 'fscore',
                  'span_f1': 'f1-measure-overall'
                 }

    def __init__(self, metric_name, **kwargs):
        """
            Check https://github.com/allenai/allennlp/tree/main/allennlp/training/metrics
            for kwargs for each metric
        """
        self._metric_name = metric_name
        self._metric = metric.metric_map[metric_name](**kwargs)

    def __call__(self, predictions, gold_targets, mask):
        self._metric(predictions, gold_targets, mask)

    def report(self):
        readout = self._metric.get_metric(reset=True)
        if isinstance(readout, float):
            return readout
        else:
            assert isinstance(readout, dict)
            if len(readout) == 1:
                return list(readout.values())[0]
            else:
                return readout[metric.return_tag[self._metric_name]]


def print_evaluation(metric_dict, epoch=None, dataset_tag=None):
    """
        Print loss and evaluation metrics on a dataset
    """
    metric_string_fields = []
    if dataset_tag is not None:
        metric_string_fields.append(dataset_tag)
    if epoch is not None:
        metric_string_fields.append('Epoch {}'.format(epoch))

    for metric_name in metric_dict:
        metric_string_fields.append('{} {:.4f}'.format(
                                    metric_name, metric_dict[metric_name]))
    metric_string = ' | '.join(metric_string_fields)

    print(metric_string, flush=True)
