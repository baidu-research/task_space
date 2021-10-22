import torch
import torch.nn as nn
from ..training import use_gpu


class simple_token_head(nn.Module):
    """
        A simple softmax/logistic classifier, or a linear regression head for
        every token in a batch.
    """
    def __init__(self, feat_dim, num_outputs, is_regression=False):
        super(simple_token_head, self).__init__()
        self.in_dim = feat_dim
        self.out_dim = num_outputs
        self.is_regression = is_regression
        if is_regression:
            self.loss = nn.MSELoss()
            self.head = nn.Linear(feat_dim, num_outputs)
        elif num_outputs > 2: # multi-class classification
            self.loss = nn.CrossEntropyLoss()
            self.head = nn.Linear(feat_dim, num_outputs)
        elif num_outputs == 2: # binary classfication
            self.loss = nn.BCEWithLogitsLoss()
            self.head = nn.Linear(feat_dim, 1)
        else:
            raise ValueError('classification problem should have at least 2 classes!')
            
    def forward(self, inputs, targets=None, mask=None):
        """
            inputs: torch.Tensor (batch, #tokens, dim)
            targets: torch.Tensor (batch, #tokens, dim) or None.
                If None,return the per-class probability for a classification problem,
                or the prediction for a regression problem
                Otherwise, return also (as the 2nd returned value) the averaged loss
            mask: torch.Tensor, boolean (batch, #tokens, dim). 
                Its True values indicate non-pad tokens
        """
        # compute loss if target provided
        predict = self.head(inputs)
        if targets is not None:
            assert mask is not None
            # squeeze for BCEloss
            loss = self.loss(predict[mask].squeeze(-1), targets[mask])

        # compute normalized probabilities if a classification problem
        if not self.is_regression:
            if self.out_dim == 2: # binary classfication
                predict = torch.sigmoid(predict) # the probability of class 1 
                predict = torch.cat((1-predict, predict), dim=-1)
            else: # multi-class classification
                predict = torch.nn.Softmax(dim=-1)(predict)

        if targets is not None:
            return predict, loss
        else:
            return predict
