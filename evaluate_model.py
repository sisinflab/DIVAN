import sys
import fuxictr_version
import logging

from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from utils.dataset_utils import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src
import gc
import argparse
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

dataset = "large"  # small, large

if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=f'./config/DIN_ebnerd_{dataset}_x1_tuner_config_01',
                        help='The config directory.')
    parser.add_argument('--expid', type=str, default=f'DIN_ebnerd_large_x1_001_b1afac3a',
                        help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')

    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform data
        feature_encoder = FeatureProcessor(**params)

        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    model_class = getattr(src, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters()  # print number of parameters used in model
    logging.info("Loading checkpoint...")
    model.to(device=model.device)
    model.load_weights(model.checkpoint)
    logging.info("Checkpoint loaded!")

    _, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate_test(valid_gen)
    del valid_gen
    gc.collect()

    test_result = {}
    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
                 .format(datetime.now().strftime('%Y%m%d-%H%M%S'),
                         ' '.join(sys.argv), experiment_id, params['dataset_id'],
                         "N.A.", print_to_list(valid_result), print_to_list(test_result)))
