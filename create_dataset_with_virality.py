# =========================================================================
# Copyright (C) 2024. FuxiCTR Authors. All rights reserved.
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
import sys
import os

# extend the sys.path to fix the import problem
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_two_up = os.path.dirname(os.path.dirname(current_dir))
sys.path.extend([parent_dir_two_up])
import logging
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
import gc
from utils.download_dataset import download_ebnerd_dataset
from utils.functions import (map_feat_id_func, tokenize_seq, impute_list_with_mean, encode_date_list,
                             compute_item_popularity_scores, get_enriched_user_history,
                             sampling_strategy_wu2019, create_binary_labels_column, exponential_decay,
                             create_inviews_vectors, compute_near_realtime_ctr)
from utils.sampling import create_test2
from fuxictr.preprocess import FeatureProcessor, build_dataset
import argparse
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
import src
from fuxictr.pytorch.dataloaders import RankDataLoader
import warnings

# AGGIUNGERE LA RAPPRESENTAZIONE DELLE INVIEW

warnings.filterwarnings("ignore")
dataset = "small"
if __name__ == '__main__':
    ''' 
    Usage: 
    python prepare_data_v1.py --size {dataset_size} --data_folder {data_path} [--test] 
                                --embedding_size [64|128|256] --embedding_type [contrastive|bert|roberta]
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str,
                        default="/Users/antodeca/PycharmProjects/FuxiCTR/RecSysChallenge2024_DIN/data/Ebnerd_small_roberta64_x1_PopPredictor",
                        help='The folder where the csv on which the PopNet was trained on')
    parser.add_argument('--config', default=f'./config/PopPredictor_ebnerd_{dataset}_x1_tuner_config_01',
                        help='The config directory.')
    parser.add_argument('--test', action="store_true", help='Use this flag to download the test set (default no)')
    parser.add_argument('--test2', action="store_true", help='Use this flag to download the test set (default no)')
    parser.add_argument('--expid', default=f"PopPredictor_ebnerd_small_x1_pop_pr_roberta64_001_2ad679ba")

    args = vars(parser.parse_args())
    #  ---Pop-Predictor-----
    config = args['config']
    dataset_folder = args['folder']  # Specify the config file of the PopPredictor
    experiment_id = args['expid']  # experiment_id is taken from the model_config.yaml
    virality_predictor_params = load_config(args['config'], experiment_id)  # Load all the params from the config file
    set_logger(virality_predictor_params)
    logging.info("Params: " + print_to_json(virality_predictor_params))
    seed_everything(seed=virality_predictor_params['seed'])
    data_dir = os.path.join(virality_predictor_params['data_root'], virality_predictor_params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if virality_predictor_params["data_format"] == "csv":
        # Build feature_map and transform data
        feature_encoder = FeatureProcessor(**virality_predictor_params)
        virality_predictor_params["train_data"], virality_predictor_params["valid_data"], virality_predictor_params[
            "test_data"] = \
            build_dataset(feature_encoder, **virality_predictor_params)
    feature_map = FeatureMap(virality_predictor_params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, virality_predictor_params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    model_class = getattr(src, virality_predictor_params['model'])
    model = model_class(feature_map, **virality_predictor_params)
    model.count_parameters()  # print number of parameters used in model
    model.to(device=model.device)
    # Insert here the checkpoint you desire
    model.load_weights(model.checkpoint)
    train_df = pl.read_csv(os.path.join(dataset_folder, "train.csv"))
    print(train_df.head())
    print("Train samples", train_df.shape)
    virality_predictor_params["shuffle"]= False
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train',
                                          **virality_predictor_params).make_iterator()
    logging.info(f"Computing the virality score with {model_class} for the Training Dataset ")
    train_df = train_df.with_columns(
        pl.Series("virality_score", model.predict(train_gen, inference=True))
    )
    train_df = train_df.select("impression_id", "article_id", "virality_score")
    train_df.write_csv(f"{dataset_folder}/train_virality.csv")
    del train_df
    gc.collect()

    valid_df = pl.read_csv(os.path.join(dataset_folder, "valid.csv"))
    print(valid_df.head())
    print("Validation samples", valid_df.shape)
    logging.info(f"Computing the virality score with {model_class} for the Validation Dataset ")
    valid_df = valid_df.with_columns(
        pl.Series("virality_score", model.predict(valid_gen, inference=True))
    )
    valid_df = valid_df.select("impression_id", "article_id", "virality_score")
    valid_df.write_csv(f"{dataset_folder}/valid_virality.csv")
    del valid_df
    gc.collect()

    if args['test']:
        test_df = pl.read_csv(os.path.join(dataset_folder, "test.csv"))
        print(test_df.head())
        print("Test samples", test_df.shape)
        test_gen = RankDataLoader(feature_map, stage='test', **virality_predictor_params).make_iterator()
        logging.info(f"Computing the virality score with {model_class} for the Test Dataset ")
        test_df = test_df.with_columns(
            pl.Series("virality_score", model.predict(test_gen, inference=True))
        )
        test_df = test_df.select("impression_id", "article_id", "virality_score")
        test_df.write_csv(f"{dataset_folder}/test_virality.csv")
        del test_df
        gc.collect()

    if args['test2']:
        test_df = pl.read_csv(os.path.join(dataset_folder, "test2.csv"))
        print(test_df.head())
        print("Test samples", test_df.shape)
        test_gen = RankDataLoader(feature_map, stage='test', **virality_predictor_params).make_iterator()
        logging.info(f"Computing the virality score with {model_class} for the Test Dataset ")
        test_df = test_df.with_columns(
            pl.Series("virality_score", model.predict(test_gen, inference=True))
        )
        test_df = test_df.select("impression_id", "article_id", "virality_score")
        test_df.write_csv(f"{dataset_folder}/test2_virality.csv")
        del test_df
        gc.collect()

    print("All done.")
