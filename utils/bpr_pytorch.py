import torch
import torch.nn.functional as F


class BPRLoss(torch.nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, y_pred, y_true, reduction='mean'):
        """
        y_true: Tensor of labels, with 1 for positive and 0 for negative.
        y_pred: Tensor of predicted scores.
        """
        # Separate positive and negative scores.
        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]

        # Calculate the BPR loss.
        loss = -(F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)))
        if reduction == 'mean':
            loss = loss.mean()
        return loss
