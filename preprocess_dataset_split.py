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
import pandas as pd

import fuxictr_version
import logging

from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor
import gc
import argparse
import os
import warnings
import multiprocessing as mp
import numpy as np
import polars as pl
import glob

warnings.filterwarnings("ignore")


def read_csv(data_path, sep=",", n_rows=None):
    logging.info("Reading file: " + data_path)
    file_names = sorted(glob.glob(data_path))
    assert len(file_names) > 0, f"Invalid data path: {data_path}"
    # Require python >= 3.8 for use polars to scan multiple csv files
    file_names = file_names[0]
    ddf = pl.scan_csv(source=file_names, separator=sep,
                      low_memory=True, n_rows=n_rows)
    return ddf


def save_npz(darray_dict, data_path):
    logging.info("Saving data to npz: " + data_path)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    np.savez(data_path, **darray_dict)


def transform_block(feature_encoder, df_block, filename):
    logging.info("Transform feature columns...")
    darray_dict = feature_encoder.transform(df_block)
    save_npz(darray_dict, os.path.join(feature_encoder.data_dir, filename))


def process_split(data_path, split_name, data_block_size=0):
    if data_path:
        idx = 0
        reader = pd.read_csv(data_path, chunksize=data_block_size)
        for chunk in reader:
            chunk = pl.DataFrame(chunk)
            df_processed = feature_encoder.preprocess(chunk)
            transform_block(feature_encoder, df_processed.to_pandas(), '{}/part_{:05d}.npz'.format(split_name, idx))

            del df_processed
            gc.collect()
            idx += 1


def transform_split(feature_encoder, train_data=None, valid_data=None, test_data=None, data_block_size=0, **kwargs):
    # Process each data split
    process_split(train_data, 'train', data_block_size)
    process_split(valid_data, 'valid', data_block_size)
    process_split(test_data, 'test', data_block_size)
    logging.info("Transform csv data to npz done.")

    # Return processed data splits
    return os.path.join(feature_encoder.data_dir, "train") if train_data else None, \
        os.path.join(feature_encoder.data_dir, "valid") if valid_data else None, \
        os.path.join(feature_encoder.data_dir, "test") if test_data else None


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=f'./config/DIN_ebnerd_demo_x1_tuner_config_01',
                        help='The config directory.')
    parser.add_argument('--expid', type=str, default=f'DIN_ebnerd_demo_x1_001_eed6a1d6',
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
