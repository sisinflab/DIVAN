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
import argparse

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    ''' 
    Usage: 
    python prepare_data_v1.py --size {dataset_size} --data_folder {data_path} [--test] 
                                --embedding_size [64|128|256] --embedding_type [contrastive|bert|roberta]
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='large', help='The size of the dataset to download')
    parser.add_argument('--data_folder', type=str, default='./data', help='The folder in which data will be stored')
    parser.add_argument('--tag', type=str, default='x1', help='The tag of the preprocessed dataset to save')
    parser.add_argument('--test', action="store_true", help='Use this flag to download the test set (default no)')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='The embedding size you want to reduce the initial embeddings')
    parser.add_argument('--embedding_type', type=str, default="contrastive",
                        help='The embedding type you want to use')

    args = vars(parser.parse_args())
    dataset_size = args['size']
    data_folder = args['data_folder']
    embedding_size = args['embedding_size']
    embedding_type = args['embedding_type']
    tag = args['tag']
    dataset_version = f"Ebnerd_{dataset_size}_{embedding_type}{embedding_size}_{tag}_PopPredictor"
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

        if args['neg_sampling']:
            create_test2()

    # Once downloaded the dataset, we have history, behaviors, articles and the embeddings
    MAX_SEQ_LEN = 50
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

    news = news.select(['article_id', 'published_time', 'last_modified_time', 'premium',
                        'article_type', 'ner_clusters', 'topics', 'category', 'subcategory',
                        'sentiment_score', 'sentiment_label'])
    news = (
        news
        .with_columns(subcat1=pl.col('subcategory').apply(lambda x: str(x[0]) if len(x) > 0 else ""))
        .collect()
    )

    news = tokenize_seq(news, 'ner_clusters', map_feat_id=True)
    news = tokenize_seq(news, 'topics', map_feat_id=True)
    news = tokenize_seq(news, 'subcategory', map_feat_id=False)
    news = map_feat_id_func(news, "sentiment_label")
    news = map_feat_id_func(news, "article_type")

    print("Save news info...")
    os.makedirs(f"{data_folder}/{dataset_version}/", exist_ok=True)
    with open(f"{data_folder}/{dataset_version}//news_info.jsonl", "w") as f:
        f.write(news.write_json(row_oriented=True, pretty=True))

    print("Preprocess behavior data...")


    def join_data(data_path):
        behavior_file = os.path.join(data_path, "behaviors.parquet")
        sample_df = pl.scan_parquet(behavior_file)
        sample_df = sample_df.select(
            ['impression_id', 'article_id', 'impression_time', 'article_ids_inview', 'article_ids_clicked',
             'next_read_time', 'next_scroll_percentage'])
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
                .collect()
            )
        sample_df = (
            sample_df
            .join(news, on='article_id', how="left"))

        sample_df = (
            sample_df
            .with_columns(
                publish_days=(pl.col('impression_time') - pl.col('published_time')).dt.days().cast(pl.Int32),
                publish_hours=(pl.col('impression_time') - pl.col('published_time')).dt.hours().cast(pl.Int32),
                impression_hour=pl.col('impression_time').dt.hour().cast(pl.Int32),
                impression_weekday=pl.col('impression_time').dt.weekday().cast(pl.Int32),
            )
            .with_columns(
                pl.col("publish_days").clip_max(3).alias("pulish_3day"),
                pl.col("publish_days").clip_max(7).alias("pulish_7day"),
                pl.col("publish_days").clip_max(30),
                pl.col("publish_hours").clip_max(24)
            )
            .drop(
                ["impression_time", "published_time", "last_modified_time", "next_scroll_percentage", "next_read_time",
                 "trigger_id", "premium"])
        )
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

    print("Preprocess pretrained embeddings...")
    image_emb_path = dataset_path + '/image_embeddings.parquet'
    image_emb_df = pl.read_parquet(image_emb_path)
    pca = PCA(n_components=embedding_size)
    image_emb = pca.fit_transform(np.array(image_emb_df["image_embedding"].to_list()))
    print("image_embedding.shape", image_emb.shape)
    item_dict = {
        "key": image_emb_df["article_id"].cast(str),
        "value": image_emb
    }
    print(f"Save image_emb_dim{embedding_size}.npz...")
    np.savez(f"{data_folder}/{dataset_version}/image_emb_dim{embedding_size}.npz", **item_dict)

    emb_path = dataset_path + f'/{embedding_type}_vector.parquet'
    emb_df = pl.read_parquet(emb_path)
    emb = pca.fit_transform(np.array(emb_df[emb_df.columns[-1]].to_list()))
    print(f"{embedding_type}_emb.shape", emb.shape)
    item_dict = {
        "key": emb_df["article_id"].cast(str),
        "value": emb
    }
    print(f"Save {embedding_type}_emb_dim{embedding_size}.npz...")
    np.savez(f"{data_folder}/{dataset_version}/{embedding_type}_emb_dim{embedding_size}.npz", **item_dict)

    print("All done.")
