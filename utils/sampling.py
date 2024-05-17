import polars as pl
import os

validation_demo = pl.scan_parquet("../data/Ebnerd_demo/validation/behaviors.parquet")
validation_len = validation_demo.collect().shape[0]
# Behaviours Large
df_behaviours = pl.scan_parquet("../validation_large/behaviors.parquet")
# Prendiamo un sample da Behaviours Large
sample = df_behaviours.collect().sample(validation_len, with_replacement=False)
# Prendiamo user_id univoci
unique_user_id = sample.select("user_id").unique().to_numpy().flatten()
# Prendiamo lista di tutti gli articoli nelle inviews
article_id_inview = sample.select("article_ids_inview").explode("article_ids_inview").rename(
    {"article_ids_inview": "article_id"})
# Leggiamo history prendendo solo utenti presenti nel sample di behaviours
df_history = pl.scan_parquet("../validation_large/history.parquet").filter(pl.col("user_id").is_in(unique_user_id))
# Prendiamo gli articoli nella history
article_id_history = df_history.collect().select("article_id_fixed").explode("article_id_fixed").rename(
    {"article_id_fixed": "article_id"})
# Concateniamo (articoli_history + article_inviews) e prendiamo quelli univoci
article_id = pl.concat([article_id_inview, article_id_history]).unique().to_numpy().flatten()
# Filtriamo articles
df_articles = pl.scan_parquet("../validation_large/articles.parquet").filter(pl.col("article_id").is_in(article_id))
# SALVATAGGIO FILE
os.makedirs("../samples_large", exist_ok=True)
df_articles.collect().write_parquet("samples_large/articles.parquet")
sample.write_parquet("samples_large/behaviors.parquet")
df_history.collect().write_parquet("samples_large/history.parquet")
