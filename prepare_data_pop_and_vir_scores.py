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

import numpy as np

# extend the sys.path to fix the import problem
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_two_up = os.path.dirname(os.path.dirname(current_dir))
sys.path.extend([parent_dir_two_up])
import polars as pl
import gc
from utils.download_dataset import download_ebnerd_dataset
from utils.functions import (compute_item_popularity_scores, exponential_decay, )
import argparse

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='small', help='The size of the dataset to download')
    parser.add_argument('--data_folder', type=str, default='./data', help='The folder in which data will be stored')
    parser.add_argument('--tag', type=str, default='x1', help='The tag of the preprocessed dataset to save')
    parser.add_argument('--test', action="store_true", help='Use this flag to download the test set (default no)')

    args = vars(parser.parse_args())
    dataset_size = args['size']
    data_folder = args['data_folder']
    tag = args['tag']
    dataset_version = f"Ebnerd_{dataset_size}_pop_and_vir_scores"
    # insert a check, if data aren't in the repository, download them
    dataset_path = os.path.join(data_folder, 'Ebnerd_' + dataset_size)
    # Check if 'Ebnerd_{dataset_size}' folder exists
    if os.path.isdir(dataset_path):
        print(f"Folder '{dataset_path}' exists.")
        # Check if 'Ebnerd_{dataset_size}' folder is empty
        if not os.listdir(dataset_path):
            print(f"Folder '{dataset_path}' is empty. Downloading the dataset...")
            # download the dataset
            if args['test']:
                print("Downloading the test set")
                download_ebnerd_dataset(dataset_size, dataset_path, dataset_path + '/train/', dataset_path + '/test/')
            else:
                print("Not Downloading the test set")
                download_ebnerd_dataset(dataset_size, dataset_path, dataset_path + '/train/')
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
        download_ebnerd_dataset(dataset_size, dataset_path, dataset_path + '/train/', dataset_path + '/test/')

    # Once downloaded the dataset, we have history, behaviors, articles and the embeddings
    train_path = dataset_path + '/train/'
    dev_path = dataset_path + '/validation/'
    test_path = dataset_path + '/test/'

    print("Preprocess news info...")
    train_news_file = os.path.join(train_path, "articles.parquet")
    train_news = pl.scan_parquet(train_news_file)

    test_news_file = os.path.join(test_path, "articles.parquet")
    test_news = pl.scan_parquet(test_news_file)

    news = pl.concat([train_news, test_news])
    news = news.unique(subset=['article_id'])
    news = news.fill_null("")

    news = news.select(['article_id', 'published_time'])

    print("Compute news popularity...")
    train_history_file = os.path.join(train_path, "history.parquet")
    valid_history_file = os.path.join(dev_path, "history.parquet")
    train_history = pl.scan_parquet(train_history_file).select(['user_id', 'article_id_fixed'])
    valid_history = pl.scan_parquet(valid_history_file).select(['user_id', 'article_id_fixed'])
    if args['test']:
        test_history_file = os.path.join(test_path, "history.parquet")
        test_history = pl.scan_parquet(test_history_file).select(['user_id', 'article_id_fixed'])

        history = pl.concat([train_history, valid_history, test_history])
        del train_history, valid_history, test_history
    else:
        history = pl.concat([train_history, valid_history])
        del train_history, valid_history

    history = history.groupby("user_id").agg(pl.col("article_id_fixed"))
    history = history.with_columns(
        pl.col("article_id_fixed").map_elements(
            lambda row: list(set([x for xs in row for x in xs]))).cast(pl.List(pl.Int32))).collect()
    history = history.fill_null("")
    gc.collect()

    # Group by user_id and aggregate the article IDs into a list
    R = history.groupby('user_id').agg(pl.col('article_id_fixed').alias('article_ids'))

    # Convert to list of np.array
    R = [np.unique(np.array(ids)) for ids in R['article_ids'].to_list()]
    popularity_scores = compute_item_popularity_scores(R)

    del history
    gc.collect()

    news = news.with_columns(
        pl.col("article_id").apply(lambda x: popularity_scores.get(x, 0.0)).alias("popularity_score"),
    ).collect()

    del R, popularity_scores
    gc.collect()

    print("Preprocess behavior data...")


    def join_data(data_path):
        behavior_file = os.path.join(data_path, "behaviors.parquet")
        sample_df = pl.scan_parquet(behavior_file)
        if "test/" in data_path:
            sample_df = (
                sample_df.rename({"article_ids_inview": "article_id"})
                .explode('article_id')
            )
            sample_df = sample_df.with_columns(
                pl.lit(None).alias("trigger_id"),
                pl.lit(0).alias("click")
            )
        else:
            sample_df = (
                sample_df.rename({"article_id": "trigger_id"})
                .rename({"article_ids_inview": "article_id"})
                .explode('article_id')
                .with_columns(click=pl.col("article_id").is_in(pl.col("article_ids_clicked")).cast(pl.Int8))
                .drop(["article_ids_clicked"])
            )
        sample_df = (
            sample_df.select("impression_id", "article_id", "click", "user_id", "impression_time").collect()
            .join(news, on='article_id', how="left")
        )
        sample_df = (
            sample_df
            .with_columns(
                publish_hours=(pl.col('impression_time') - pl.col('published_time')).dt.hours().cast(pl.Int32),
            )
            .with_columns(
                freshness_decay=pl.col('publish_hours').apply(exponential_decay)
            )
            .with_columns(
                virality_score=(pl.col('popularity_score') * (pl.col('freshness_decay')))
            )
        )

        sample_df = sample_df.select("impression_id", "user_id", "article_id", "click", "popularity_score",
                                     "virality_score")
        print(sample_df.columns)
        return sample_df


    if os.path.isdir(f"{data_folder}/{dataset_version}"):
        print(f"Folder '{data_folder}/{dataset_version}' exists.")
    else:
        os.makedirs(f"{data_folder}/{dataset_version}")
        print(f"Folder '{data_folder}/{dataset_version}' has been created.")

    train_df = join_data(train_path)
    print(train_df.head())
    print("Train samples", train_df.shape)
    train_df.write_csv(f"{data_folder}/{dataset_version}/train.csv")
    del train_df
    gc.collect()

    valid_df = join_data(dev_path)
    print(valid_df.head())
    print("Validation samples", valid_df.shape)
    valid_df.write_csv(f"{data_folder}/{dataset_version}/valid.csv")
    del valid_df
    gc.collect()

    if args['test']:
        test_df = join_data(test_path)
        print(test_df.head())
        print("Test samples", test_df.shape)
        test_df.write_csv(f"{data_folder}/{dataset_version}/test.csv")
        del test_df
        gc.collect()
