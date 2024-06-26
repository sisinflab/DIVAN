# =========================================================================
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


import sys
import fuxictr_version
import logging

from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from utils.dataset_utils import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, transform
import src
import gc
import argparse
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def transform_split(feature_encoder, train_data=None, valid_data=None, test_data=None, data_block_size=0, **kwargs):
    # fit and transform train_ddf
    if train_data:
        train_ddf = feature_encoder.read_csv(train_data, **kwargs)
        train_ddf = feature_encoder.preprocess(train_ddf)
        # feature_encoder.fit(train_ddf, **kwargs)
        transform(feature_encoder, train_ddf, 'train', block_size=data_block_size)
        del train_ddf
        gc.collect()

    if valid_data:
        valid_ddf = feature_encoder.read_csv(valid_data, **kwargs)
        valid_ddf = feature_encoder.preprocess(valid_ddf)
        transform(feature_encoder, valid_ddf, 'valid', block_size=data_block_size)
        del valid_ddf
        gc.collect()

    # Transfrom test_ddf
    if test_data:
        test_ddf = feature_encoder.read_csv(test_data, **kwargs)
        test_ddf = feature_encoder.preprocess(test_ddf)
        transform(feature_encoder, test_ddf, 'test', block_size=data_block_size)
        del test_ddf
        gc.collect()
    logging.info("Transform csv data to npz done.")

    # Return processed data splits
    return os.path.join(feature_encoder.data_dir, "train") if train_data else None, \
        os.path.join(feature_encoder.data_dir, "valid") if valid_data else None, \
        os.path.join(feature_encoder.data_dir, "test") if (
            test_data) else None


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=f'./config/DIVAN_ebnerd_demo_x1_tuner_config_01',
                        help='The config directory.')
    parser.add_argument('--expid', type=str, default=f'DIVAN_ebnerd_demo_x1_001_43881344',
                        help='The experiment id to run.')
    parser.add_argument('--split', type=str, default=f'all',
                        help='The split to preprocess [train|valid|test|all].')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform data
        feature_encoder = FeatureProcessor(**params)
        feature_encoder = feature_encoder.load_pickle(feature_encoder.pickle_file)
        if args['split'] == 'train':
            params["valid_data"] = None
            params["test_data"] = None
        elif args['split'] == 'valid':
            params["train_data"] = None
            params["test_data"] = None
        elif args['split'] == 'test':
            params["valid_data"] = None
            params["train_data"] = None
        else:
            pass

        transform_split(feature_encoder, **params)
