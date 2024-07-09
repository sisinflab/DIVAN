import torch
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
from collections import OrderedDict

def evaluate_metrics(y_true, y_pred, metrics, group_id=None):
    return_dict = OrderedDict()
    group_metrics = []
    for metric in metrics:
        if metric in ['logloss', 'binary_crossentropy']:
            return_dict[metric] = log_loss(y_true.numpy(), y_pred.numpy(), eps=1e-7)
        elif metric == 'AUC':
            return_dict[metric] = roc_auc_score(y_true.numpy(), y_pred.numpy())
        elif metric in ["gAUC", "avgAUC", "MRR"] or metric.startswith("NDCG"):
            return_dict[metric] = 0
            group_metrics.append(metric)
        else:
            raise ValueError(f"metric={metric} not supported.")

    if len(group_metrics) > 0:
        assert group_id is not None, "group_index is required."
        metric_funcs = []
        for metric in group_metrics:
            try:
                metric_funcs.append(eval(metric))
            except:
                raise NotImplementedError(f'metrics={metric} not implemented.')

        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)
        group_id = torch.tensor(group_id)

        unique_groups = torch.unique(group_id)
        results = [evaluate_block(y_true[group_id == group], y_pred[group_id == group], metric_funcs) for group in
                   unique_groups]
        results = torch.tensor(results)

        sum_results = results.sum(dim=0)
        average_result = (sum_results[:, 0] / sum_results[:, 1]).tolist()
        return_dict.update(dict(zip(group_metrics, average_result)))

    return return_dict


def evaluate_block(y_true, y_pred, metric_funcs):
    res_list = []
    for fn in metric_funcs:
        v = fn(y_true.numpy(), y_pred.numpy())
        if isinstance(v, tuple):
            res_list.append(v)
        else:  # add group weight
            res_list.append((v, 1))
    return res_list


def avgAUC(y_true, y_pred):
    """ avgAUC used in MIND news recommendation """
    if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
        auc = roc_auc_score(y_true, y_pred)
        return (auc, 1)
    else:  # in case all negatives or all positives for a group
        return (0, 0)


def gAUC(y_true, y_pred):
    """ gAUC defined in DIN paper """
    if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
        auc = roc_auc_score(y_true, y_pred)
        n_samples = len(y_true)
        return (auc * n_samples, n_samples)
    else:  # in case all negatives or all positives for a group
        return (0, 0)


def MRR(y_true, y_pred):
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    mrr = np.sum(rr_score) / (np.sum(y_true) + 1e-12)
    return mrr


class NDCG(object):
    """Normalized discounted cumulative gain metric."""

    def __init__(self, k=1):
        self.topk = k

    def dcg_score(self, y_true, y_pred):
        order = np.argsort(y_pred)[::-1]
        y_true = np.take(y_true, order[:self.topk])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def __call__(self, y_true, y_pred):
        idcg = self.dcg_score(y_true, y_true)
        dcg = self.dcg_score(y_true, y_pred)
        return dcg / (idcg + 1e-12)
