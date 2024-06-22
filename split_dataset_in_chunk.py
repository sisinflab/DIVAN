import argparse

import polars as pl
import os
from utils.download_dataset import download_ebnerd_dataset
from utils.sampling import create_test2
from utils.functions import copy_folder, create_chunks

if __name__ == '__main__':
    ''' 
    Usage: 
    python prepare_data_v1.py --size {dataset_size} --data_folder {data_path} [--test] 
                                --embedding_size [64|128|256] --embedding_type [contrastive|bert|roberta]
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='large', help='The size of the dataset to download')
    parser.add_argument('--data_folder', type=str, default='./data', help='The folder in which data will be stored')
    parser.add_argument('--test', action="store_true", help='Use this flag to download the test set (default no)')
    parser.add_argument('--embedding_type', type=str, default='roberta',
                        help='The embedding type you want to use')
    parser.add_argument('--neg_sampling', action="store_true", help='Use this flag to perform negative sampling')
    parser.add_argument('--num_users', default=20000, action="store_true", help='The chunk dimension')

    args = vars(parser.parse_args())
    dataset_size = args['size']
    data_folder = args['data_folder']
    embedding_type = args['embedding_type']
    num_users = args['num_users']
    # insert a check, if data aren't in the repository, download them
    dataset_path = os.path.join(data_folder, 'Ebnerd_' + dataset_size)
    output_path = os.path.join(data_folder, f"Ebnerd_{dataset_size}_chunk")
    # Check if 'Ebnerd_{dataset_size}' folder exists
    if os.path.isdir(dataset_path):
        print(f"Folder '{dataset_path}' exists.")
        # Check if 'Ebnerd_{dataset_size}' folder is empty
        if not os.listdir(dataset_path):
            print(f"Folder '{dataset_path}' is empty. Downloading the dataset...")
            # download the dataset
            if args['test']:
                print("Downloading the test set")
                download_ebnerd_dataset(dataset_size, embedding_type, dataset_path, dataset_path + '/train/',
                                        dataset_path + '/test/')
            else:
                print("Not Downloading the test set")
                download_ebnerd_dataset(dataset_size, embedding_type, dataset_path, dataset_path + '/train/')
        else:
            print(f"Folder '{dataset_path}' is not empty. The dataset is already downloaded")
            # end, we will not download anything
    else:
        print(f"Folder '{dataset_path}' does nost exist. Creating it now.")

        # Create the 'ebnerd_demo' folder
        os.makedirs(dataset_path)
        print(f"Folder '{dataset_path}' has been created.")
        # now we will download the dataset here
        print("Downloading the data set")
        download_ebnerd_dataset(dataset_size, embedding_type, dataset_path, dataset_path + '/train/',
                                dataset_path + '/test/')

        if args['neg_sampling']:
            copy_folder(os.path.join(dataset_path, "validation"), os.path.join(dataset_path, "test2"))
    create_chunks(dataset_path, output_path, num_users)
