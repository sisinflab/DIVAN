# =========================================================================
# Copyright (C) 2024. FuxiCTR Authors. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
import torch.nn as nn
import numpy as np
from fuxictr.metrics import evaluate_metrics
from fuxictr.pytorch.torch_utils import get_device
import logging
import os, sys
from tqdm import tqdm


class PopularRanker(nn.Module):
    def __init__(self,
                 feature_map,
                 model_id="PopularRanker",
                 gpu=-1,
                 **kwargs):
        super(PopularRanker, self).__init__()
        self.validation_metrics = kwargs["metrics"]
        self._verbose = kwargs["verbose"]
        self.feature_map = feature_map
        self.model_id = model_id
        self.device = get_device(gpu)

    def get_inputs(self, inputs, feature_source=None):
        if feature_source and type(feature_source) == str:
            feature_source = [feature_source]
        X_dict = dict()
        for feature, spec in self.feature_map.features.items():
            if (feature_source is not None) and (spec["source"] not in feature_source):
                continue
            if spec["type"] == "meta":
                continue
            X_dict[feature] = inputs[:, self.feature_map.get_column_index(feature)].to(self.device)
        return X_dict

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        y = inputs[:, self.feature_map.get_column_index(labels[0])].to(self.device)
        return y.float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[:, self.feature_map.get_column_index(self.feature_map.group_id)]

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        y_pred = X['popularity_score']
        return_dict = {"y_pred": y_pred}
        return return_dict

    def evaluate(self, data_generator, metrics=None):
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        return evaluate_metrics(y_true, y_pred, metrics, group_id)