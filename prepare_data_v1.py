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
                             create_inviews_vectors)
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
    parser.add_argument('--size', type=str, default='demo', help='The size of the dataset to download')
    parser.add_argument('--data_folder', type=str, default='./data', help='The folder in which data will be stored')
    parser.add_argument('--tag', type=str, default='x1', help='The tag of the preprocessed dataset to save')
    parser.add_argument('--test', action="store_true", help='Use this flag to download the test set (default no)')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='The embedding size you want to reduce the initial embeddings')
    parser.add_argument('--embedding_types', type=list, default=['contrastive', 'bert', 'roberta'],
                        help='The embedding type you want to use')
    parser.add_argument('--neg_sampling', action="store_true", help='Use this flag to perform negative sampling')

    args = vars(parser.parse_args())
    dataset_size = args['size']
    data_folder = args['data_folder']
    embedding_size = args['embedding_size']
    embedding_types = args['embedding_types']
    tag = args['tag']
    dataset_version = f"Ebnerd_{dataset_size}_{embedding_size}_{tag}"
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
            create_test2(dataset_path)

    # Once downloaded the dataset, we have history, behaviors, articles and the embeddings
    MAX_SEQ_LEN = 100
    train_path = dataset_path + '/train/'
    dev_path = dataset_path + '/validation/'
    test_path = dataset_path + '/test/'

    print("Preprocess news info...")
    train_news_file = os.path.join(train_path, "articles.parquet")
    train_news = pl.scan_parquet(train_news_file)

    test_news_file = os.path.join(test_path, "articles.parquet")
    test_news = pl.scan_parquet(test_news_file)

    if args['neg_sampling']:
        test2_path = dataset_path + '/test2/'
        if not os.path.isdir(test2_path):
            create_test2(dataset_path)
        test2_news_file = os.path.join(test2_path, "articles.parquet")
        test2_news = pl.scan_parquet(test2_news_file)
        news = pl.concat([train_news, test_news, test2_news])
        del test2_news
    else:
        news = pl.concat([train_news, test_news])
    
    del train_news, test_news
    gc.collect()

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

    news2cat = dict(zip(news["article_id"].cast(str), news["category"].cast(str)))
    news2subcat = dict(zip(news["article_id"].cast(str), news["subcat1"].cast(str)))
    news = tokenize_seq(news, 'ner_clusters', map_feat_id=True)
    news = tokenize_seq(news, 'topics', map_feat_id=True)
    news = tokenize_seq(news, 'subcategory', map_feat_id=False)
    news = map_feat_id_func(news, "sentiment_label")
    news = map_feat_id_func(news, "article_type")
    news2sentiment = dict(zip(news["article_id"].cast(str), news["sentiment_label"]))
    news2type = dict(zip(news["article_id"].cast(str), news["article_type"]))

    news = (
        news
        .with_columns(topic1=pl.col('topics').apply(lambda x: str(x.split("^")[0]) if len(x.split("^")) > 0 else ""))
        .with_columns(topic2=pl.col('topics').apply(lambda x: str(x.split("^")[1]) if len(x.split("^")) > 1 else ""))
        .with_columns(topic3=pl.col('topics').apply(lambda x: str(x.split("^")[2]) if len(x.split("^")) > 2 else ""))
    )
    news2topic1 = dict(zip(news["article_id"].cast(str), news["topic1"].cast(str)))
    news2topic2 = dict(zip(news["article_id"].cast(str), news["topic2"].cast(str)))
    news2topic3 = dict(zip(news["article_id"].cast(str), news["topic3"].cast(str)))


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

    train_behaviors_file = os.path.join(train_path, "behaviors.parquet")
    valid_behaviors_file = os.path.join(dev_path, "behaviors.parquet")
    train_behaviors = pl.scan_parquet(train_behaviors_file)
    valid_behaviors = pl.scan_parquet(valid_behaviors_file)

    behaviors = pl.concat([train_behaviors, valid_behaviors])
    behaviors = behaviors.unique(subset=['impression_id'])
    behaviors = behaviors.fill_null("").collect()

    R = get_enriched_user_history(behaviors, history)
    popularity_scores = compute_item_popularity_scores(R)

    del train_behaviors, valid_behaviors, history
    gc.collect()

    news = news.with_columns(
        pl.col("article_id").apply(lambda x: popularity_scores.get(x, 0.0)).alias("popularity_score"),
    )

    news2pop = dict(zip(news["article_id"].cast(str), news["popularity_score"].cast(str)))

    del R, popularity_scores
    gc.collect()

    print(news.head())
    print("Save news info...")
    os.makedirs(f"{data_folder}/{dataset_version}/", exist_ok=True)
    with open(f"{data_folder}/{dataset_version}//news_info.jsonl", "w") as f:
        f.write(news.write_json(row_oriented=True, pretty=True))

    print("Preprocess behavior data...")


    def join_data(data_path):
        history_file = os.path.join(data_path, "history.parquet")
        history_df = pl.scan_parquet(history_file)
        history_df = history_df.rename({"article_id_fixed": "hist_id",
                                        "read_time_fixed": "hist_read_time",
                                        "impression_time_fixed": "hist_time",
                                        "scroll_percentage_fixed": "hist_scroll_percent"})

        # missing imputation of hist_scroll_percent, hist_read_time, hist_time
        history_df = history_df.with_columns(
            pl.col("hist_scroll_percent").apply(impute_list_with_mean),
            pl.col("hist_read_time").apply(impute_list_with_mean),
            pl.col("hist_time").apply(encode_date_list)
        )

        history_df = tokenize_seq(history_df, 'hist_id', map_feat_id=False, max_seq_length=MAX_SEQ_LEN)
        history_df = tokenize_seq(history_df, 'hist_read_time', map_feat_id=False, max_seq_length=MAX_SEQ_LEN)
        history_df = tokenize_seq(history_df, 'hist_scroll_percent', map_feat_id=False, max_seq_length=MAX_SEQ_LEN)
        history_df = tokenize_seq(history_df, 'hist_time', map_feat_id=False, max_seq_length=MAX_SEQ_LEN)

        history_df = history_df.with_columns(
            pl.col("hist_id").apply(lambda x: "^".join([news2cat.get(i, "") for i in x.split("^")])).alias("hist_cat"),
            pl.col("hist_id").apply(lambda x: "^".join([news2subcat.get(i, "") for i in x.split("^")])).alias(
                "hist_subcat1"),
            pl.col("hist_id").apply(lambda x: "^".join([news2topic1.get(i, "") for i in x.split("^")])).alias(
                "hist_topic1"),
            pl.col("hist_id").apply(lambda x: "^".join([news2topic2.get(i, "") for i in x.split("^")])).alias(
                "hist_topic2"),
            pl.col("hist_id").apply(lambda x: "^".join([news2topic3.get(i, "") for i in x.split("^")])).alias(
                "hist_topic3"),
            pl.col("hist_id").apply(lambda x: "^".join([news2sentiment.get(i, "") for i in x.split("^")])).alias(
                "hist_sentiment"),
            pl.col("hist_id").apply(lambda x: "^".join([news2type.get(i, "") for i in x.split("^")])).alias(
                "hist_type"),
            pl.col('hist_id').apply(lambda x: "^".join([news2pop.get(i, "0") for i in x.split("^")])).alias(
                "hist_pop")
        )
        history_df = history_df.collect()
        behavior_file = os.path.join(data_path, "behaviors.parquet")
        sample_df = pl.scan_parquet(behavior_file)
        sample_df.drop("gender", "postcode", "age")
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
            if args['neg_sampling']:
                print("Performing negative sampling...")
                if "test2" in data_path:
                    sample_df = (
                        sample_df.rename({"article_id": "trigger_id"})
                        .rename({"article_ids_inview": "article_id"})
                        .explode('article_id')
                        .with_columns(click=pl.col("article_id").is_in(pl.col("article_ids_clicked")).cast(pl.Int8))
                        .drop(["article_ids_clicked"])
                        .collect()
                    )
                else:
                    sample_df = (
                        sample_df.rename({"article_id": "trigger_id"})
                        .collect()
                        .pipe(sampling_strategy_wu2019, npratio=14, shuffle=True, clicked_col="article_ids_clicked",
                              inview_col="article_ids_inview", with_replacement=True, seed=123)
                        .with_columns(
                            pl.col("impression_id").cast(pl.String) + pl.col("article_ids_clicked").cum_count().cast(
                                pl.Int32).cast(
                                pl.String))
                        .with_columns(pl.col("impression_id").cast(pl.Int64))
                        .pipe(create_binary_labels_column, clicked_col="article_ids_clicked",
                              inview_col="article_ids_inview")
                        .drop("labels")
                        .rename({"article_ids_inview": "article_id"})
                        .explode("article_id")
                        .with_columns(click=pl.col("article_id").is_in(pl.col("article_ids_clicked")).cast(pl.Int8))
                        .drop("article_ids_clicked")
                        .with_columns(pl.col("article_id").cast(pl.Int32))
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
            .join(news, on='article_id', how="left")
            .join(history_df, on='user_id', how="left"))

        sample_df = (
            sample_df
            .with_columns(
                publish_days=(pl.col('impression_time') - pl.col('published_time')).dt.days().cast(pl.Int32),
                publish_hours=(pl.col('impression_time') - pl.col('published_time')).dt.hours().cast(pl.Int32),
                impression_hour=pl.col('impression_time').dt.hour().cast(pl.Int32),
                impression_weekday=pl.col('impression_time').dt.weekday().cast(pl.Int32),
            )
            .with_columns(
                freshness_decay=pl.col('publish_hours').apply(exponential_decay)
            )
            .with_columns(
                virality_score=(pl.col('popularity_score') * (pl.col('freshness_decay')))
            )
            .with_columns(
                pl.col("publish_days").clip_max(3).alias("pulish_3day"),
                pl.col("publish_days").clip_max(7).alias("pulish_7day"),
                pl.col("publish_days").clip_max(30),
                pl.col("publish_hours").clip_max(24)
            )
            .drop(
                ["impression_time", "published_time", "last_modified_time", "next_scroll_percentage", "next_read_time"])
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
    gc.collect()

    valid_df = join_data(dev_path)
    print(valid_df.head())
    print("Validation samples", valid_df.shape)
    valid_df.write_csv(f"{data_folder}/{dataset_version}/valid.csv")
    del valid_df
    gc.collect()

    if args['neg_sampling']:
        test2_df = join_data(test2_path)
        print(test2_df.head())
        print("Test samples", test2_df.shape)
        test2_df.write_csv(f"{data_folder}/{dataset_version}/test2.csv")
        del test2_df
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
    del image_emb_df, image_emb, item_dict
    gc.collect()

    for embedding_type in embedding_types:
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
        del emb, item_dict
        gc.collect()

        print("Create a representation of the inviews")
        if args['test']:
            behavior_file_test = os.path.join(test_path, "behaviors.parquet")
            behavior_df_test = pl.scan_parquet(behavior_file_test)

            behaviors = pl.concat([behaviors, behavior_df_test])
            behaviors = behaviors.unique(subset=['impression_id'])
            del behavior_df_test
            gc.collect()

        impr_ids, inviews_vectors = create_inviews_vectors(behaviors, emb_df)
        inviews_emb = pca.fit_transform(inviews_vectors)
        print("inviews_emb.shape", inviews_emb.shape)
        item_dict = {
            "key": impr_ids.cast(str),
            "value": inviews_emb
        }
        print(f"Save inviews_emb_dim{embedding_size}.npz...")
        np.savez(f"{data_folder}/{dataset_version}/inviews_{embedding_type}_emb_dim{embedding_size}.npz", **item_dict)
        del emb_df, behaviors, impr_ids, inviews_vectors, inviews_emb, item_dict
        gc.collect()
    print("All done.")
