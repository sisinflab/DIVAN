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
import gc
from utils.download_dataset import download_ebnerd_dataset
from utils.functions import (compute_item_popularity_scores, get_enriched_user_history, clean_dataframe)

import warnings

warnings.filterwarnings("ignore")
dataset_size = 'small'  # demo, small, large

# Download the datasets and put them to the following folders
train_path = "./train/"
dev_path = "./validation/"
test_path = "./test/"

image_emb_path = "image_embeddings.parquet"
contrast_emb_path = "contrastive_vector.parquet"

dataset_version = f"Ebnerd_{dataset_size}_pop"
MAX_SEQ_LEN = 50

# download_ebnerd_dataset(dataset_size=dataset_size, train_path=train_path, val_path=dev_path, test_path=test_path)

print("Preprocess news info...")
train_news_file = os.path.join(train_path, "articles.parquet")
train_news = pl.scan_parquet(train_news_file)

test_news_file = os.path.join(test_path, "articles.parquet")
test_news = pl.scan_parquet(test_news_file)

news = pl.concat([train_news, test_news])
news = news.unique(subset=['article_id'])
news = news.fill_null("")

news = news.select('article_id')

print("Compute news popularity...")
train_history_file = os.path.join(train_path, "history.parquet")
train_history = pl.scan_parquet(train_history_file).select("user_id", "article_id_fixed")

valid_history_file = os.path.join(dev_path, "history.parquet")
valid_history = pl.scan_parquet(valid_history_file).select("user_id", "article_id_fixed")

test_history_file = os.path.join(test_path, "history.parquet")
test_history = pl.scan_parquet(test_history_file).select("user_id", "article_id_fixed")

# history = train_history.join(valid_history, on="user_id", how="outer")
history = pl.concat([train_history, valid_history, test_history])
history = history.groupby("user_id").agg(pl.col("article_id_fixed"))
history = history.with_columns(
    pl.col("article_id_fixed").map_elements(
        lambda row: list(set([x for xs in row for x in xs]))).cast(pl.List(pl.Int32))).collect()
history = history.fill_null("")

del train_history, valid_history, test_history
gc.collect()

train_behaviors_file = os.path.join(train_path, "behaviors.parquet")
train_behaviors = pl.scan_parquet(train_behaviors_file)

valid_behaviors_file = os.path.join(dev_path, "behaviors.parquet")
valid_behaviors = pl.scan_parquet(valid_behaviors_file)

behaviors = pl.concat([train_behaviors, valid_behaviors])

behaviors = behaviors.unique(subset=['impression_id'])
behaviors = behaviors.fill_null("").collect()

del train_behaviors, valid_behaviors
gc.collect()

R = get_enriched_user_history(behaviors, history)
popularity_scores = compute_item_popularity_scores(R)

del history, behaviors, R
gc.collect()

news = news.with_columns(
    pl.col("article_id").apply(lambda x: popularity_scores.get(x, 0.0)).alias("popularity_score")
).collect()

del train_news, test_news
gc.collect()

print(news.head())
print("Save news info...")
os.makedirs(dataset_version, exist_ok=True)
with open(f"./{dataset_version}/news_info.jsonl", "w") as f:
    f.write(news.write_json(row_oriented=True, pretty=True))

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
        sample_df.select("impression_id", "article_id", "click", "user_id").collect()
        .join(news, on='article_id', how="left")
    )
    print(sample_df.columns)
    return sample_df


train_df = join_data(train_path)
print(train_df.head())
print("Train samples", train_df.shape)
train_df.write_csv(f"./{dataset_version}/train.csv")
del train_df
gc.collect()

valid_df = join_data(dev_path)
print(valid_df.head())
print("Validation samples", valid_df.shape)
valid_df.write_csv(f"./{dataset_version}/valid.csv")
del valid_df
gc.collect()

test_df = join_data(test_path)
print(test_df.head())
print("Test samples", test_df.shape)
test_df.write_csv(f"./{dataset_version}/test.csv")
del test_df
gc.collect()

# remove unuseful files and directories
# os.remove('train/behaviors.parquet')
# os.remove('train/history.parquet')
# os.remove('train/articles.parquet')
# os.removedirs("train")
# os.remove('test/behaviors.parquet')
# os.remove('test/history.parquet')
# os.remove('test/articles.parquet')
# os.removedirs("test")
# os.remove('validation/behaviors.parquet')
# os.remove('validation/history.parquet')
# os.removedirs("validation")
# os.remove('test2/behaviors.parquet')
# os.remove('test2/history.parquet')
# os.removedirs("test2")
# os.remove("contrastive_vector.parquet")
# os.remove("image_embeddings.parquet")

print("All done.")
