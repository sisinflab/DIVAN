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
import numpy as np
import pandas as pd
import torch
from torch import nn
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, DIN_Attention, Dice
from utils.bpr import BPRLoss
from fuxictr.pytorch.torch_utils import get_optimizer, get_loss
import logging
from tqdm import tqdm
import sys


class DIN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DIN",
                 gpu=-1,
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0,
                 batch_norm=False,
                 din_target_field=[("item_id", "cate_id")],
                 din_sequence_field=[("click_history", "cate_history")],
                 din_use_softmax=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DIN, self).__init__(feature_map,
                                      model_id=model_id,
                                      gpu=gpu,
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        if not isinstance(din_target_field, list):
            din_target_field = [din_target_field]
        self.din_target_field = din_target_field
        if not isinstance(din_sequence_field, list):
            din_sequence_field = [din_sequence_field]
        self.din_sequence_field = din_sequence_field
        assert len(self.din_target_field) == len(self.din_sequence_field), \
            "len(din_target_field) != len(din_sequence_field)"
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.attention_layers = nn.ModuleList(
            [DIN_Attention(embedding_dim * len(target_field) if type(target_field) == tuple \
                               else embedding_dim,
                           attention_units=attention_hidden_units,
                           hidden_activations=attention_hidden_activations,
                           output_activation=attention_output_activation,
                           dropout_rate=attention_dropout,
                           use_softmax=din_use_softmax)
             for target_field in self.din_target_field])
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
        self.loss_name = kwargs["loss"]

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field,
                                                                 self.din_sequence_field)):
            target_emb = self.get_embedding(target_field, feature_emb_dict)
            sequence_emb = self.get_embedding(sequence_field, feature_emb_dict)
            seq_field = list(flatten([sequence_field]))[0]  # flatten nested list to pick the first sequence field
            mask = X[seq_field].long() != 0  # padding_idx = 0 required
            pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask)
            for field, field_emb in zip(list(flatten([sequence_field])),
                                        pooling_emb.split(self.embedding_dim, dim=-1)):
                feature_emb_dict[field] = field_emb
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)

        if self.loss_name == 'bpr':
            y_true = y_true.data.cpu().numpy().reshape(-1)
            y_pred = return_dict["y_pred"].data.cpu().numpy().reshape(-1)
            group_id = self.get_group_id(batch_data).data.cpu().numpy().reshape(-1)
            return_dict_grouped, y_true_grouped = self.get_scores_grouped_by_impression(group_id, y_true, y_pred)
            loss = self.compute_loss(return_dict_grouped, y_true_grouped)
        else:
            loss = self.compute_loss(return_dict, y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss

    @staticmethod
    def get_embedding(field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        if loss == 'bpr':
            self.loss_fn = BPRLoss(max_positives=1)  # max_positives = npratio + 1
        else:
            self.loss_fn = get_loss(loss)

    def get_scores_grouped_by_impression(self, group_id, y_true, y_pred):
        score_df = pd.DataFrame({"group_index": group_id,
                                 "y_true": y_true,
                                 "y_pred": y_pred})

        idxs = []
        y_true_list = []
        y_pred_list = []

        for idx, df in score_df.groupby("group_index"):
            idxs.append(idx)
            y_true_list.append(df['y_true'].values)
            y_pred_list.append(df['y_pred'].values)

        return_dict = {'y_pred': torch.Tensor(np.array(y_pred_list))}

        return return_dict, torch.Tensor(np.array(y_true_list))

    def evaluate(self, data_generator, metrics=None):
        val_loss = 0
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred_list = []
            y_true_list = []
            group_id_list = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred = return_dict["y_pred"].data.cpu().numpy().reshape(-1)
                y_true = self.get_labels(batch_data).data.cpu().numpy().reshape(-1)
                group_id = self.get_group_id(batch_data).numpy().reshape(-1)

                # compute loss on validation
                if self.loss_name == 'bpr':
                    return_dict_grouped, y_true_grouped = self.get_scores_grouped_by_impression(group_id,
                                                                                                y_true,
                                                                                                y_pred)
                    loss = self.compute_loss(return_dict_grouped, y_true_grouped)
                else:
                    loss = self.compute_loss(return_dict, self.get_labels(batch_data))

                val_loss += loss.item()

                y_pred_list.extend(y_pred)
                y_true_list.extend(y_true)
                if self.feature_map.group_id is not None:
                    group_id_list.extend(group_id)

            y_pred = np.array(y_pred_list, np.float64)
            y_true = np.array(y_true_list, np.float64)
            group_id = np.array(group_id_list) if len(group_id) > 0 else None

            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
            logging.info("Val loss: {:.6f}".format(val_loss / len(data_generator)))
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs
