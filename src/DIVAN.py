import numpy as np
import torch
from torch import nn
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, DIN_Attention, Dice
from src.PopNet import PopNet
from src.Gate import Gate
from utils.bpr_pytorch import BPRLoss
from fuxictr.pytorch.torch_utils import get_optimizer, get_loss
import logging
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter


class DIVAN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DIVAN",
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
                 recency_field=[("publish_hours")],
                 din_use_softmax=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 gate_hidden_units=[100],
                 gate_dropout=0,
                 network_recency_hidden_dims=[100],
                 network_recency_dropout_rate=0,
                 network_content_hidden_dims=[100],
                 network_content_dropout_rate=0,
                 **kwargs):
        super(DIVAN, self).__init__(feature_map,
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
        if not isinstance(recency_field, list):
            recency_field = [recency_field]
        self.recency_field = recency_field
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

        self.gate = Gate(
            input_dim=embedding_dim * len([i for el in self.din_sequence_field for i in el]),
            hidden_dims=gate_hidden_units,
            dropout_rate=gate_dropout
        )

        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)

        self.popnet = PopNet(
            network_recency_input_dim=embedding_dim * len([i for el in self.recency_field for i in el]),
            network_recency_hidden_dims=network_recency_hidden_dims,
            network_recency_dropout_rate=network_recency_dropout_rate,
            network_content_input_dim=embedding_dim * len([i for el in self.din_target_field for i in el]),
            network_content_hidden_dims=network_content_hidden_dims,
            network_content_dropout_rate=network_content_dropout_rate,
            gate_hidden_units=gate_hidden_units,
            gate_dropout=gate_dropout)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
        self.loss_name = kwargs["loss"]

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        labels = self.get_labels(inputs)
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

        # Get user-specific alpha
        user_emb = self.embedding_layer.dict2tensor(
            {i: feature_emb_dict[i] for el in self.din_sequence_field for i in el}, flatten_emb=True)
        alpha = self.gate(user_emb)

        # predict news popularity
        news_recency_emb = self.embedding_layer.dict2tensor(
            {i: feature_emb_dict[i] for el in self.recency_field for i in el}, flatten_emb=True)
        news_content_emb = self.embedding_layer.dict2tensor(
            {i: feature_emb_dict[i] for el in self.din_target_field for i in el}, flatten_emb=True)
        y_pred_pop = self.popnet(news_recency_emb, news_content_emb)

        # predict din scores
        y_pred_din = self.dnn(feature_emb)

        # Combine the two prediction scores with user-specific parameters
        y_pred = alpha * y_pred_din + (1 - alpha) * y_pred_pop

        if self._total_steps % self._eval_steps == 0:
            return_dict = {"y_pred": y_pred.float(),
                           "positive_y_pred": y_pred[labels == 1].mean(),
                           "negative_y_pred": y_pred[labels == 0].mean(),
                           "positive_y_pred_din": y_pred_din[labels == 1].mean(),
                           "negative_y_pred_din": y_pred_din[labels == 0].mean(),
                           "positive_y_pred_pop": y_pred_pop[labels == 1].mean(),
                           "negative_y_pred_pop": y_pred_pop[labels == 0].mean(),
                           "alpha": alpha.mean()}
        else:
            return_dict = {"y_pred": self.output_activation(y_pred).float()}
        return return_dict

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)

        if self.loss_name == 'bpr':
            group_id = self.get_group_id(batch_data)
            return_dict_grouped, y_true_grouped = self.get_scores_grouped_by_impression(group_id, y_true,
                                                                                        return_dict["y_pred"])
            loss = self.compute_loss(return_dict_grouped, y_true_grouped)
        else:
            loss = self.compute_loss(return_dict, y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss, return_dict

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0

        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss, return_dict = self.train_step(batch_data)

            train_loss += loss.item()

            if self._total_steps % self._eval_steps == 0:
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                self.writer.add_scalar("Train_Loss_per_Epoch", train_loss / self._eval_steps, self._epoch_index)
                self.writer.add_scalars("mean_comb_scores", {
                    "mean_positive_comb_scores": return_dict['positive_y_pred'].mean() / self._eval_steps,
                    "mean_negative_comb_scores": return_dict['negative_y_pred'] / self._eval_steps
                }, self._epoch_index)
                self.writer.add_scalars("mean_din_scores", {
                    "mean_positive_din_scores": return_dict['positive_y_pred_din'] / self._eval_steps,
                    "mean_negative_din_scores": return_dict['negative_y_pred_din'] / self._eval_steps
                }, self._epoch_index)
                self.writer.add_scalars("mean_virality_scores", {
                    "mean_positive_pop_scores": return_dict['positive_y_pred_pop'] / self._eval_steps,
                    "mean_negative_pop_scores": return_dict['negative_y_pred_pop'] / self._eval_steps
                }, self._epoch_index)
                self.writer.add_scalar("alpha", return_dict['alpha'], self._epoch_index)
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break

    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.writer = SummaryWriter(comment=self.model_id)
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        for epoch in range(epochs):
            self._epoch_index = epoch
            logging.info("************ Epoch={} start ************".format(self._epoch_index + 1))
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.writer.close()
        self.load_weights(self.checkpoint)

    def eval_step(self):
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)
        self.train()

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
            self.loss_fn = BPRLoss()
        else:
            self.loss_fn = get_loss(loss)

    def get_scores_grouped_by_impression(self, group_id, y_true, y_pred):
        unique_groups = torch.unique(group_id)
        y_true_list = []
        y_pred_list = []

        for group in unique_groups:
            mask = (group_id == group)
            y_true_list.append(y_true[mask])
            y_pred_list.append(y_pred[mask])

        return_dict = {'y_pred': torch.stack(y_pred_list)}

        return return_dict, torch.stack(y_true_list)

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
                y_true = self.get_labels(batch_data)
                group_id = self.get_group_id(batch_data)

                # compute loss on validation
                if self.loss_name == 'bpr':
                    return_dict_grouped, y_true_grouped = self.get_scores_grouped_by_impression(group_id,
                                                                                                y_true,
                                                                                                return_dict["y_pred"])
                    loss = self.compute_loss(return_dict_grouped, y_true_grouped)
                else:
                    loss = self.compute_loss(return_dict, self.get_labels(batch_data))

                val_loss += loss.item()

                y_pred_list.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true_list.extend(y_true.data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id_list.extend(group_id.numpy().reshape(-1))

            y_pred = np.array(y_pred_list, np.float64)
            y_true = np.array(y_true_list, np.float64)
            group_id = np.array(group_id_list) if len(group_id) > 0 else None
            self.writer.add_scalar("Validation_Loss_per_epoch", val_loss / len(data_generator), self._epoch_index)

            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
            logging.info("Val loss: {:.6f}".format(val_loss / len(data_generator)))
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            self.writer.add_scalar("avgAUC_per_epoch", val_logs['avgAUC'], self._epoch_index)
            return val_logs

    def evaluate_test(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
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
