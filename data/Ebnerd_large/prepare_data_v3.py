import sys
import os

# extend the sys.path to fix the import problem
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_two_up = os.path.dirname(os.path.dirname(current_dir))
sys.path.extend([parent_dir_two_up])

import polars as pl
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
import gc
from utils.download_dataset import download_ebnerd_dataset
from utils.functions import (map_feat_id_func, tokenize_seq, impute_list_with_mean, encode_date_list,
                             compute_item_popularity_scores, get_enriched_user_history)
from utils.sampling import create_test_for_large
from utils.polars_utils import slice_join_dataframes
import warnings

warnings.filterwarnings("ignore")
dataset_size = 'large'  # demo, small, large

# Download the datasets and put them to the following folders
train_path = "./train/"
dev_path = "./validation/"
test_path = "./test/"
test2_path = "./test2/"

image_emb_path = "image_embeddings.parquet"
contrast_emb_path = "contrastive_vector.parquet"

dataset_version = f"Ebnerd_{dataset_size}_x1"
MAX_SEQ_LEN = 50

download_ebnerd_dataset(dataset_size=dataset_size, train_path=train_path, val_path=dev_path, test_path=test_path)
create_test_for_large()

print("Preprocess news info...")
train_news_file = os.path.join(train_path, "articles.parquet")
test_news_file = os.path.join(test_path, "articles.parquet")
test2_news_file = os.path.join(test2_path, "articles.parquet")

news_files = [train_news_file, test_news_file, test2_news_file]
news_dfs = [pl.scan_parquet(f) for f in news_files]

news = pl.concat(news_dfs).unique(subset=['article_id']).fill_null("")
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

print("Compute news popularity...")

history_files = [os.path.join(train_path, "history.parquet"),
                 os.path.join(dev_path, "history.parquet"),
                 os.path.join(test_path, "history.parquet"),
                 os.path.join(test2_path, "history.parquet")]

history_dfs = [pl.scan_parquet(f) for f in history_files]

history = pl.concat(history_dfs).unique(subset=['user_id']).fill_null("")
del history_dfs
gc.collect()

train_behaviors_file = os.path.join(train_path, "behaviors.parquet")
valid_behaviors_file = os.path.join(dev_path, "behaviors.parquet")
test2_behaviors_file = os.path.join(test2_path, "behaviors.parquet")
behavior_files = [train_behaviors_file, valid_behaviors_file, test2_behaviors_file]

behaviors_dfs = [pl.scan_parquet(f) for f in behavior_files]

behaviors = pl.concat(behaviors_dfs).unique(subset=['impression_id']).fill_null("")
del behaviors_dfs
gc.collect()

R = get_enriched_user_history(behaviors, history, chunk_size=5000)
popularity_scores = compute_item_popularity_scores(R)

del history, R
gc.collect()

news = news.with_columns(
    pl.col("article_id").apply(lambda x: popularity_scores.get(x, 0)).alias("popularity_score")
)

print(news.head())
print("Save news info...")
os.makedirs(dataset_version, exist_ok=True)
news.write_json(f"./{dataset_version}/news_info.jsonl", row_oriented=True, pretty=True)

print("Preprocess behavior data...")


def join_data(data_path, chunk_size=500000):
    history_file = os.path.join(data_path, "history.parquet")
    behavior_file = os.path.join(data_path, "behaviors.parquet")

    history_df = pl.scan_parquet(history_file).select([
        pl.col("article_id_fixed").alias("hist_id"),
        pl.col("read_time_fixed").alias("hist_read_time"),
        pl.col("impression_time_fixed").alias("hist_time"),
        pl.col("scroll_percentage_fixed").alias("hist_scroll_percent"),
        pl.col("user_id")
    ])

    # Missing imputation and encoding
    print("Missing imputation and encoding...")
    history_df = history_df.with_columns([
        pl.col("hist_scroll_percent").apply(impute_list_with_mean).alias("hist_scroll_percent"),
        pl.col("hist_read_time").apply(impute_list_with_mean).alias("hist_read_time"),
        pl.col("hist_time").apply(encode_date_list).alias("hist_time"),

    ])

    # Tokenize sequences
    print("Tokenize sequences...")
    for col in ["hist_id", "hist_read_time", "hist_scroll_percent", "hist_time"]:
        history_df = tokenize_seq(history_df, col, map_feat_id=False, max_seq_length=MAX_SEQ_LEN)

    # Adding new columns based on hist_id in chunks
    print("Adding new columns based on hist_id...")

    history_chunks = []
    for chunk in history_df.collect().iter_slices(chunk_size):
        chunk = chunk.with_columns([
            pl.col("hist_id").apply(lambda x: "^".join([news2cat.get(i, "") for i in x.split("^")])).alias("hist_cat"),
            pl.col("hist_id").apply(lambda x: "^".join([news2subcat.get(i, "") for i in x.split("^")])).alias(
                "hist_subcat1"),
            pl.col("hist_id").apply(lambda x: "^".join([news2sentiment.get(i, "") for i in x.split("^")])).alias(
                "hist_sentiment"),
            pl.col("hist_id").apply(lambda x: "^".join([news2type.get(i, "") for i in x.split("^")])).alias("hist_type")
        ])

        history_chunks.append(chunk)
        del chunk
        gc.collect()
    history_df = pl.concat(history_chunks)
    del history_chunks
    gc.collect()

    # Preprocess behavior data in chunks
    print("Preprocess behavior data in chunks...")
    sample_df = pl.read_parquet(behavior_file).drop(["gender", "postcode", "age"])

    if "test/" in data_path:
        sample_df = (
            sample_df.rename({"article_ids_inview": "article_id"})
            .explode("article_id")
            .with_columns([
                pl.lit(None).alias("trigger_id"),
                pl.lit(0).alias("click")
            ]))
    else:
        sample_df = (
            sample_df.rename({"article_id": "trigger_id"})
            .rename({"article_ids_inview": "article_id"})
            .explode("article_id")
            .with_columns(click=pl.col("article_id").is_in(pl.col("article_ids_clicked")).cast(pl.Int8))
            .drop(["article_ids_clicked"]))

    sample_df = (
        news.pipe(slice_join_dataframes, df2=sample_df, on='article_id', how="left", chunk_size=chunk_size)
        .pipe(slice_join_dataframes, history_df, on='user_id', how="left", chunk_size=chunk_size)
        .with_columns(
            publish_days=(pl.col('impression_time') - pl.col('published_time')).dt.days().cast(pl.Int32),
            publish_hours=(pl.col('impression_time') - pl.col('published_time')).dt.hours().cast(pl.Int32),
            impression_hour=pl.col('impression_time').dt.hour().cast(pl.Int32),
            impression_weekday=pl.col('impression_time').dt.weekday().cast(pl.Int32)
        )
        .with_columns(
            pl.col("publish_days").clip_max(3).alias("publish_3day"),
            pl.col("publish_days").clip_max(7).alias("publish_7day"),
            pl.col("publish_days").clip_max(30),
            pl.col("publish_hours").clip_max(24)
        )
        .drop(
            ["impression_time", "published_time", "last_modified_time", "next_scroll_percentage", "next_read_time"])
    )
    return sample_df


print("\nWriting train...")
train_df = join_data(train_path)
print(train_df.head())
print("Train samples", train_df.shape)
train_df.write_csv(f"./{dataset_version}/train.csv")
del train_df
gc.collect()

print("\nWriting validation...")
valid_df = join_data(dev_path)
print(valid_df.head())
print("Validation samples", valid_df.shape)
valid_df.write_csv(f"./{dataset_version}/valid.csv")
del valid_df
gc.collect()

print("\nWriting test2...")
test2_df = join_data(test2_path)
print(test2_df.head())
print("Test2 samples", test2_df.shape)
test2_df.write_csv(f"./{dataset_version}/test2.csv")
del test2_df
gc.collect()

print("\nWriting test...")
test_df = join_data(test_path)
print(test_df.head())
print("Test samples", test_df.shape)
test_df.write_csv(f"./{dataset_version}/test.csv")
del test_df
gc.collect()

print("\nPreprocess pretrained embeddings...")
image_emb_df = pl.read_parquet(image_emb_path)
pca = PCA(n_components=64)
image_emb = pca.fit_transform(np.array(image_emb_df["image_embedding"].to_list()))
print("image_embedding.shape", image_emb.shape)
item_dict = {
    "key": image_emb_df["article_id"].cast(str),
    "value": image_emb
}
print("Save image_emb_dim64.npz...")
np.savez(f"./{dataset_version}/image_emb_dim64.npz", **item_dict)

contrast_emb_df = pl.read_parquet(contrast_emb_path)
contrast_emb = pca.fit_transform(np.array(contrast_emb_df["contrastive_vector"].to_list()))
print("contrast_emb.shape", contrast_emb.shape)
item_dict = {
    "key": contrast_emb_df["article_id"].cast(str),
    "value": contrast_emb
}
print("Save contrast_emb_dim64.npz...")
np.savez(f"./{dataset_version}/contrast_emb_dim64.npz", **item_dict)


def create_inviews_vectors(behavior_df):
    inviews_ids = behavior_df.select('impression_id', 'article_ids_inview').collect()
    inviews_vectors = []
    for inview in inviews_ids['article_ids_inview'].to_list():
        inview_vectors = []
        for item_id in inview:
            inview_vectors.append(
                contrast_emb_df.filter(pl.col('article_id') == item_id)['contrastive_vector'].to_list())
        inviews_vectors.append(np.array(inview_vectors).mean(axis=0))
    return inviews_ids["impression_id"], np.array(inviews_vectors).squeeze(axis=1)


print("\nCreate a representation of the inviews")
behavior_file_test = os.path.join(test_path, "behaviors.parquet")
behavior_df_test = pl.scan_parquet(behavior_file_test)

behaviors = pl.concat([behaviors, behavior_df_test])
behaviors = behaviors.unique(subset=['impression_id'])

impr_ids, inviews_vectors = create_inviews_vectors(behaviors)
inviews_emb = pca.fit_transform(inviews_vectors)
print("inviews_emb.shape", inviews_emb.shape)
item_dict = {
    "key": impr_ids.cast(str),
    "value": inviews_emb
}
print("Save inviews_emb_dim64.npz...")
np.savez(f"./{dataset_version}/inviews_emb_dim64.npz", **item_dict)

# remove unuseful files and directories
os.remove('train/behaviors.parquet')
os.remove('train/history.parquet')
os.remove('train/articles.parquet')
os.removedirs("train")
os.remove('test/behaviors.parquet')
os.remove('test/history.parquet')
os.remove('test/articles.parquet')
os.removedirs("test")
os.remove('validation/behaviors.parquet')
os.remove('validation/history.parquet')
os.removedirs("validation")
os.remove('test2/behaviors.parquet')
os.remove('test2/history.parquet')
os.removedirs("test2")
os.remove("contrastive_vector.parquet")
os.remove("image_embeddings.parquet")

print("All done.")
