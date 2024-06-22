import polars as pl
import os
import zipfile
import requests
import io


def create_test2(path):
    dataset_url = f"https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip"
    response_dataset = requests.get(dataset_url)
    with zipfile.ZipFile(io.BytesIO(response_dataset.content)) as zip_ref:
        zip_ref.extract("articles.parquet", path=f"{path}/Ebnerd_large/")
        zip_ref.extract("validation/history.parquet", path=f"{path}/Ebnerd_large/")
        zip_ref.extract("validation/behaviors.parquet", path=f"{path}/Ebnerd_large/")

    validation_demo = pl.scan_parquet(f"{path}/validation/behaviors.parquet")
    validation_len = validation_demo.collect().shape[0]
    # Behaviours Large
    df_behaviours = pl.scan_parquet(f"{path}/Ebnerd_large/validation/behaviors.parquet")
    # Prendiamo un sample da Behaviours Large
    sample = df_behaviours.collect().sample(validation_len, with_replacement=False)
    # Prendiamo user_id univoci
    unique_user_id = sample.select("user_id").unique().to_numpy().flatten()
    # Prendiamo lista di tutti gli articoli nelle inviews
    article_id_inview = sample.select("article_ids_inview").explode("article_ids_inview").rename(
        {"article_ids_inview": "article_id"})
    # Leggiamo history prendendo solo utenti presenti nel sample di behaviours
    df_history = pl.scan_parquet(f"{path}/Ebnerd_large/validation/history.parquet").filter(
        pl.col("user_id").is_in(unique_user_id))
    # Prendiamo gli articoli nella history
    article_id_history = df_history.collect().select("article_id_fixed").explode("article_id_fixed").rename(
        {"article_id_fixed": "article_id"})
    # Concateniamo (articoli_history + article_inviews) e prendiamo quelli univoci
    article_id = pl.concat([article_id_inview, article_id_history]).unique().to_numpy().flatten()
    # Filtriamo articles
    df_articles = pl.scan_parquet(f"{path}/Ebnerd_large/articles.parquet").filter(
        pl.col("article_id").is_in(article_id))
    # SALVATAGGIO FILE
    os.makedirs(f"{path}/test2", exist_ok=True)
    df_articles.collect().write_parquet(f"{path}/test2/articles.parquet")
    sample.write_parquet(f"{path}/test2/behaviors.parquet")
    df_history.collect().write_parquet(f"{path}/test2/history.parquet")
    os.remove(f"{path}/Ebnerd_large/articles.parquet")
    os.remove(f"{path}/Ebnerd_large/validation/behaviors.parquet")
    os.remove(f"{path}/Ebnerd_large/validation/history.parquet")
    os.removedirs(f"{path}/Ebnerd_large/validation/")


def split_dataset_in_chunks(dataset_path):
    validation_demo = pl.scan_parquet(f"{dataset_path}/validation/behaviors.parquet")
    validation_len = validation_demo.collect().shape[0]
    # Behaviours Large
    df_behaviours = pl.scan_parquet(f"{dataset_path}/Ebnerd_large/validation/behaviors.parquet")
    # Prendiamo un sample da Behaviours Large
    df_behaviours = df_behaviours.collect().sample(frac=1.0, with_replacement=False)
    sample = df_behaviours.collect().sample(validation_len, with_replacement=False)
    # Prendiamo user_id univoci
    unique_user_id = sample.select("user_id").unique().to_numpy().flatten()
    # Prendiamo lista di tutti gli articoli nelle inviews
    article_id_inview = sample.select("article_ids_inview").explode("article_ids_inview").rename(
        {"article_ids_inview": "article_id"})
    # Leggiamo history prendendo solo utenti presenti nel sample di behaviours
    df_history = pl.scan_parquet(f"{path}/Ebnerd_large/validation/history.parquet").filter(
        pl.col("user_id").is_in(unique_user_id))
    # Prendiamo gli articoli nella history
    article_id_history = df_history.collect().select("article_id_fixed").explode("article_id_fixed").rename(
        {"article_id_fixed": "article_id"})
    # Concateniamo (articoli_history + article_inviews) e prendiamo quelli univoci
    article_id = pl.concat([article_id_inview, article_id_history]).unique().to_numpy().flatten()
    # Filtriamo articles
    df_articles = pl.scan_parquet(f"{path}/Ebnerd_large/articles.parquet").filter(
        pl.col("article_id").is_in(article_id))
    # SALVATAGGIO FILE
    os.makedirs(f"{path}/test2", exist_ok=True)
    df_articles.collect().write_parquet(f"{path}/test2/articles.parquet")
    sample.write_parquet(f"{path}/test2/behaviors.parquet")
    df_history.collect().write_parquet(f"{path}/test2/history.parquet")
    os.remove(f"{path}/Ebnerd_large/articles.parquet")
    os.remove(f"{path}/Ebnerd_large/validation/behaviors.parquet")
    os.remove(f"{path}/Ebnerd_large/validation/history.parquet")
    os.removedirs(f"{path}/Ebnerd_large/validation/")


def create_test_for_large():
    print("Creating test set for Ebnerd large")
    dataset_url = f"https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip"
    response_dataset = requests.get(dataset_url)
    with zipfile.ZipFile(io.BytesIO(response_dataset.content)) as zip_ref:
        zip_ref.extract("validation/history.parquet", path="./Ebnerd_small/")
        zip_ref.extract("validation/behaviors.parquet", path="./Ebnerd_small/")
        zip_ref.extract("articles.parquet", path="./Ebnerd_small/validation")
    os.rename("./Ebnerd_small/validation", "./test2/")
    print("Test set created!")
