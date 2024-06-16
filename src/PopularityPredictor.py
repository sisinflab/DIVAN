import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import sys
import logging
from torch.utils.tensorboard import SummaryWriter


class PopPredictor(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="Pop_Predictor",
                 gpu=-1,
                 learning_rate=1e-3,
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_dim=10,
                 embedding_regularizer=None,  # Controllare se serve
                 net_regularizer=None,  # Controllare se serve
                 **kwargs):
        super(PopPredictor, self).__init__(feature_map,
                                           model_id=model_id,
                                           gpu=gpu,
                                           embedding_regularizer=embedding_regularizer,
                                           net_regularizer=net_regularizer,
                                           **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)

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

    def predict(self, data_generator, inference=False):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data, inference=inference)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            return y_pred

    def forward(self, inputs, inference=False):
        X = self.get_inputs(inputs)
        labels = self.get_labels(inputs)
        feature_emb_dict = self.embedding_layer(X)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        y_pred = self.dnn(feature_emb)
        # Questo if è stato aggiuto perchè quando voglio usare un modello preallenato non chiamo la fit e quind
        # self.eval_steps è settato a none,self.total_steps non esiste
        if inference:
            return_dict = {"y_pred": y_pred.float()}
        else:
            if self._total_steps % self._eval_steps == 0:
                return_dict = {"y_pred": y_pred.float(),
                               "positive_y_pred_din": y_pred[labels == 1].mean(),
                               "negative_y_pred_din": y_pred[labels == 0].mean()}
            else:
                return_dict = {"y_pred": y_pred.float()}
        return return_dict

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
            loss = self.train_step(batch_data)
            train_loss += loss.item()
            if self._total_steps % self._eval_steps == 0:
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                self.writer.add_scalar("Train_Loss_per_Epoch", train_loss / self._eval_steps, self._epoch_index)
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break

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
