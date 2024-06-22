from typing import Any, Iterable
from pathlib import Path
from tqdm import tqdm
import warnings
import datetime
import inspect
from collections import Counter
from typing import Iterable
import numpy as np
from pandas.core.common import flatten
import polars as pl
import gc
from datetime import timedelta
import os
import shutil

from utils.polars_utils import (
    _check_columns_in_df,
    drop_nulls_from_list,
    generate_unique_name,
    shuffle_list_column,
)
from utils.polars_utils import slice_join_dataframes

from utils.constants import (
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_KNOWN_USER_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL
)
from utils.python_utils import create_lookup_dict


def reorder_lists(df: pl.DataFrame, article_col: str, label_col: str):
    def reorder(article_ids_inview, labels):
        combined = list(zip(labels, article_ids_inview))
        sorted_combined = sorted(combined, key=lambda x: -x[0])
        sorted_labels, sorted_article_ids = zip(*sorted_combined)
        return list(sorted_article_ids)

        # Apply the function using Polars apply method

    # reordered_data = pl.apply([df[article_col], df[label_col]], reorder)
    df = df.with_columns(
        pl.struct([article_col, label_col]).map_elements(lambda x: reorder(x[article_col], x[label_col])).alias(
            "article_ids_ordered"))

    # Apply the function using Polars apply method
    return df


def create_binary_labels_column(
        df: pl.DataFrame,
        shuffle: bool = True,
        seed: int = None,
        clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
        inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
        label_col: str = DEFAULT_LABELS_COL,
) -> pl.DataFrame:
    """Creates a new column in a DataFrame containing binary labels indicating
    whether each article ID in the "article_ids" column is present in the corresponding
    "list_destination" column.

    Args:
        df (pl.DataFrame): The input DataFrame.

    Returns:
        pl.DataFrame: A new DataFrame with an additional "labels" column.

    Examples:
    >>> from RecSysChallenge2024_DIN.utils.constants import (
            DEFAULT_CLICKED_ARTICLES_COL,
            DEFAULT_INVIEW_ARTICLES_COL,
            DEFAULT_LABELS_COL,
        )
    >>> df = pl.DataFrame(
            {
                DEFAULT_INVIEW_ARTICLES_COL: [[1, 2, 3], [4, 5, 6], [7, 8]],
                DEFAULT_CLICKED_ARTICLES_COL: [[2, 3, 4], [3, 5], None],
            }
        )
    >>> create_binary_labels_column(df)
        shape: (3, 3)
        ┌────────────────────┬─────────────────────┬───────────┐
        │ article_ids_inview ┆ article_ids_clicked ┆ labels    │
        │ ---                ┆ ---                 ┆ ---       │
        │ list[i64]          ┆ list[i64]           ┆ list[i8]  │
        ╞════════════════════╪═════════════════════╪═══════════╡
        │ [1, 2, 3]          ┆ [2, 3, 4]           ┆ [0, 1, 1] │
        │ [4, 5, 6]          ┆ [3, 5]              ┆ [0, 1, 0] │
        │ [7, 8]             ┆ null                ┆ [0, 0]    │
        └────────────────────┴─────────────────────┴───────────┘
    >>> create_binary_labels_column(df.lazy(), shuffle=True, seed=123).collect()
        shape: (3, 3)
        ┌────────────────────┬─────────────────────┬───────────┐
        │ article_ids_inview ┆ article_ids_clicked ┆ labels    │
        │ ---                ┆ ---                 ┆ ---       │
        │ list[i64]          ┆ list[i64]           ┆ list[i8]  │
        ╞════════════════════╪═════════════════════╪═══════════╡
        │ [3, 1, 2]          ┆ [2, 3, 4]           ┆ [1, 0, 1] │
        │ [5, 6, 4]          ┆ [3, 5]              ┆ [1, 0, 0] │
        │ [7, 8]             ┆ null                ┆ [0, 0]    │
        └────────────────────┴─────────────────────┴───────────┘
    Test_:
    >>> assert create_binary_labels_column(df, shuffle=False)[DEFAULT_LABELS_COL].to_list() == [
            [0, 1, 1],
            [0, 1, 0],
            [0, 0],
        ]
    >>> assert create_binary_labels_column(df, shuffle=True)[DEFAULT_LABELS_COL].list.sum().to_list() == [
            2,
            1,
            0,
        ]
    """
    _check_columns_in_df(df, [inview_col, clicked_col])
    _COLUMNS = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMNS, "_groupby_id")

    df = df.with_row_index(GROUPBY_ID)

    if shuffle:
        df = shuffle_list_column(df, column=inview_col, seed=seed)

    df_labels = (
        df.explode(inview_col)
        .with_columns(
            pl.col(inview_col).is_in(pl.col(clicked_col)).cast(pl.Int8).alias(label_col)
        )
        .group_by(GROUPBY_ID)
        .agg(label_col)
    )
    return (
        df.join(df_labels, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMNS + [label_col])
    )


def create_user_id_to_int_mapping(
        df: pl.DataFrame, user_col: str = DEFAULT_USER_COL, value_str: str = "id"
):
    return create_lookup_dict(
        df.select(pl.col(user_col).unique()).with_row_index(value_str),
        key=user_col,
        value=value_str,
    )


def create_chunks(dataset_path, output_path, num_users):
    # Behaviours Large
    print(f"Slicing for train")
    df_history_train = pl.scan_parquet(f"{dataset_path}/train/history.parquet")
    df_behaviours_train = pl.scan_parquet(f"{dataset_path}/train/behaviors.parquet")
    df_articles_train = pl.scan_parquet(f"{dataset_path}/train/articles.parquet")

    df_history_val = pl.scan_parquet(f"{dataset_path}/validation/history.parquet")
    df_behaviours_val = pl.scan_parquet(f"{dataset_path}/validation/behaviors.parquet")
    df_articles_val = pl.scan_parquet(f"{dataset_path}/validation/articles.parquet")
    df_history_train = df_history_train.collect().sample(fraction=1.0, with_replacement=False)
    df_history_val = df_history_val.collect().sample(fraction=1.0, with_replacement=False)
    for idx, chunk in enumerate(df_history_train.iter_slices(num_users)):
        create_chunk(idx, chunk, df_behaviours_train, df_articles_train, output_path, "train")
    for idx, chunk in enumerate(df_history_val.iter_slices(num_users)):
        create_chunk(idx, chunk, df_behaviours_val, df_articles_val, output_path, "validation")
        copy_folder(f"{dataset_path}/test2", os.path.join(output_path, "test2"))
        copy_file(f"{dataset_path}/roberta_vector.parquet",
                  os.path.join(output_path, "roberta_vector.parquet"))
        copy_file(f"{dataset_path}/image_embeddings.parquet",
                  os.path.join(output_path, "image_embeddings.parquet"))


def copy_file(src, dst):
    # Check if the source file exists
    if not os.path.isfile(src):
        print(f"Source file '{src}' does not exist.")
        return

    # Copy the file
    shutil.copy2(src, dst)
    print(f"Copied '{src}' to '{dst}'")


def create_chunk(idx, chunk, df_behaviors, df_articles, output_path, split):
    chunk_user_ids = chunk.select("user_id").to_numpy().flatten()
    behaviours_chunk = df_behaviors.filter(pl.col("user_id").is_in(chunk_user_ids))
    article_id_inview = behaviours_chunk.select("article_ids_inview").explode("article_ids_inview").rename(
        {"article_ids_inview": "article_id"}).collect()
    article_id_history = chunk.select("article_id_fixed").explode("article_id_fixed").rename(
        {"article_id_fixed": "article_id"})
    article_id = pl.concat([article_id_inview, article_id_history]).unique().to_numpy().flatten()
    # Da fare solo per train e test2
    df_articles = df_articles.filter(pl.col("article_id").is_in(article_id))
    current_chunk_path = os.path.join(output_path, f"chunk{idx}", split)
    os.makedirs(current_chunk_path, exist_ok=True)
    chunk.write_parquet(os.path.join(current_chunk_path, "history.parquet"))
    behaviours_chunk.collect().write_parquet(os.path.join(current_chunk_path, "behaviors.parquet"))
    df_articles.collect().write_parquet(os.path.join(current_chunk_path, "articles.parquet"))


def copy_folder(src, dst):
    # Check if the source directory exists
    if not os.path.exists(src):
        print(f"Source directory '{src}' does not exist.")
        return

    # Check if the destination directory already exists
    if os.path.exists(dst):
        print(f"Destination directory '{dst}' already exists.")
        return

    # Copy the entire directory tree
    shutil.copytree(src, dst)
    print(f"Copied '{src}' to '{dst}'")


def filter_minimum_negative_samples(
        df,
        n: int,
        inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
        clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
) -> pl.DataFrame:
    """
    >>> from RecSysChallenge2024_DIN.utils.constants import DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_INVIEW_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                DEFAULT_INVIEW_ARTICLES_COL: [[1, 2, 3], [1], [1, 2, 3]],
                DEFAULT_CLICKED_ARTICLES_COL: [[1], [1], [1, 2]],
            }
        )
    >>> filter_minimum_negative_samples(df, n=1)
        shape: (2, 2)
        ┌────────────────────┬─────────────────────┐
        │ article_ids_inview ┆ article_ids_clicked │
        │ ---                ┆ ---                 │
        │ list[i64]          ┆ list[i64]           │
        ╞════════════════════╪═════════════════════╡
        │ [1, 2, 3]          ┆ [1]                 │
        │ [1, 2, 3]          ┆ [1, 2]              │
        └────────────────────┴─────────────────────┘
    >>> filter_minimum_negative_samples(df, n=2)
        shape: (3, 2)
        ┌─────────────┬──────────────────┐
        │ article_ids ┆ list_destination │
        │ ---         ┆ ---              │
        │ list[i64]   ┆ list[i64]        │
        ╞═════════════╪══════════════════╡
        │ [1, 2, 3]   ┆ [1]              │
        └─────────────┴──────────────────┘
    """
    return (
        df.filter((pl.col(inview_col).list.len() - pl.col(clicked_col).list.len()) >= n)
        if n is not None and n > 0
        else df
    )


def filter_read_times(df, n: int, column: str) -> pl.DataFrame:
    """
    Use this to set the cutoff for 'read_time' and 'next_read_time'
    """
    return (
        df.filter(pl.col(column) >= n)
        if column in df and n is not None and n > 0
        else df
    )


def unique_article_ids_in_behaviors(
        df: pl.DataFrame,
        col: str = "ids",
        item_col: str = DEFAULT_ARTICLE_ID_COL,
        inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
        clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
) -> pl.Series:
    """
    Examples:
        >>> df = pl.DataFrame({
                DEFAULT_ARTICLE_ID_COL: [1, 2, 3, 4],
                DEFAULT_INVIEW_ARTICLES_COL: [[2, 3], [1, 4], [4], [1, 2, 3]],
                DEFAULT_CLICKED_ARTICLES_COL: [[], [2], [3, 4], [1]],
            })
        >>> unique_article_ids_in_behaviors(df).sort()
            [
                1
                2
                3
                4
            ]
    """
    df = df.lazy()
    return (
        pl.concat(
            (
                df.select(pl.col(item_col).unique().alias(col)),
                df.select(pl.col(inview_col).explode().unique().alias(col)),
                df.select(pl.col(clicked_col).explode().unique().alias(col)),
            )
        )
        .drop_nulls()
        .unique()
        .collect()
    ).to_series()


def add_known_user_column(
        df: pl.DataFrame,
        known_users: Iterable[int],
        user_col: str = DEFAULT_USER_COL,
        known_user_col: str = DEFAULT_KNOWN_USER_COL,
) -> pl.DataFrame:
    """
    Adds a new column to the DataFrame indicating whether the user ID is in the list of known users.
    Args:
        df: A Polars DataFrame object.
        known_users: An iterable of integers representing the known user IDs.
    Returns:
        A new Polars DataFrame with an additional column 'is_known_user' containing a boolean value
        indicating whether the user ID is in the list of known users.
    Examples:
        >>> df = pl.DataFrame({'user_id': [1, 2, 3, 4]})
        >>> add_known_user_column(df, [2, 4])
            shape: (4, 2)
            ┌─────────┬───────────────┐
            │ user_id ┆ is_known_user │
            │ ---     ┆ ---           │
            │ i64     ┆ bool          │
            ╞═════════╪═══════════════╡
            │ 1       ┆ false         │
            │ 2       ┆ true          │
            │ 3       ┆ false         │
            │ 4       ┆ true          │
            └─────────┴───────────────┘
    """
    return df.with_columns(pl.col(user_col).is_in(known_users).alias(known_user_col))


def sample_article_ids(
        df: pl.DataFrame,
        n: int,
        with_replacement: bool = False,
        seed: int = None,
        inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
) -> pl.DataFrame:
    """
    Randomly sample article IDs from each row of a DataFrame with or without replacement

    Args:
        df: A polars DataFrame containing the column of article IDs to be sampled.
        n: The number of article IDs to sample from each list.
        with_replacement: A boolean indicating whether to sample with replacement.
            Default is False.
        seed: An optional seed to use for the random number generator.

    Returns:
        A new polars DataFrame with the same columns as `df`, but with the article
        IDs in the specified column replaced by a list of `n` sampled article IDs.

    Examples:
    >>> from RecSysChallenge2024_DIN.utils.constants import DEFAULT_INVIEW_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "clicked": [
                    [1],
                    [4, 5],
                    [7, 8, 9],
                ],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    ["A", "B", "C"],
                    ["D", "E", "F"],
                    ["G", "H", "I"],
                ],
                "col" : [
                    ["h"],
                    ["e"],
                    ["y"]
                ]
            }
        )
    >>> print(df)
        shape: (3, 3)
        ┌──────────────────┬─────────────────┬───────────┐
        │ list_destination ┆ article_ids     ┆ col       │
        │ ---              ┆ ---             ┆ ---       │
        │ list[i64]        ┆ list[str]       ┆ list[str] │
        ╞══════════════════╪═════════════════╪═══════════╡
        │ [1]              ┆ ["A", "B", "C"] ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "E", "F"] ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "H", "I"] ┆ ["y"]     │
        └──────────────────┴─────────────────┴───────────┘
    >>> sample_article_ids(df, n=2, seed=42)
        shape: (3, 3)
        ┌──────────────────┬─────────────┬───────────┐
        │ list_destination ┆ article_ids ┆ col       │
        │ ---              ┆ ---         ┆ ---       │
        │ list[i64]        ┆ list[str]   ┆ list[str] │
        ╞══════════════════╪═════════════╪═══════════╡
        │ [1]              ┆ ["A", "C"]  ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "F"]  ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "I"]  ┆ ["y"]     │
        └──────────────────┴─────────────┴───────────┘
    >>> sample_article_ids(df.lazy(), n=4, with_replacement=True, seed=42).collect()
        shape: (3, 3)
        ┌──────────────────┬───────────────────┬───────────┐
        │ list_destination ┆ article_ids       ┆ col       │
        │ ---              ┆ ---               ┆ ---       │
        │ list[i64]        ┆ list[str]         ┆ list[str] │
        ╞══════════════════╪═══════════════════╪═══════════╡
        │ [1]              ┆ ["A", "A", … "C"] ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "D", … "F"] ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "G", … "I"] ┆ ["y"]     │
        └──────────────────┴───────────────────┴───────────┘
    """
    _check_columns_in_df(df, [inview_col])
    _COLUMNS = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMNS, "_groupby_id")
    df = df.with_row_count(name=GROUPBY_ID)

    df_ = (
        df.explode(inview_col)
        .group_by(GROUPBY_ID)
        .agg(
            pl.col(inview_col).sample(n=n, with_replacement=with_replacement, seed=seed)
        )
    )
    return (
        df.drop(inview_col)
        .join(df_, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMNS)
    )


def remove_positives_from_inview(
        df: pl.DataFrame,
        inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
        clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
):
    """Removes all positive article IDs from a DataFrame column containing inview articles and another column containing
    clicked articles. Only negative article IDs (i.e., those that appear in the inview articles column but not in the
    clicked articles column) are retained.

    Args:
        df (pl.DataFrame): A DataFrame with columns containing inview articles and clicked articles.

    Returns:
        pl.DataFrame: A new DataFrame with only negative article IDs retained.

    Examples:
    >>> from RecSysChallenge2024_DIN.utils.constants import DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "user_id": [1, 1, 2],
                DEFAULT_CLICKED_ARTICLES_COL: [
                    [1, 2],
                    [1],
                    [3],
                ],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                ],
            }
        )
    >>> remove_positives_from_inview(df)
        shape: (3, 3)
        ┌─────────┬─────────────────────┬────────────────────┐
        │ user_id ┆ article_ids_clicked ┆ article_ids_inview │
        │ ---     ┆ ---                 ┆ ---                │
        │ i64     ┆ list[i64]           ┆ list[i64]          │
        ╞═════════╪═════════════════════╪════════════════════╡
        │ 1       ┆ [1, 2]              ┆ [3]                │
        │ 1       ┆ [1]                 ┆ [2, 3]             │
        │ 2       ┆ [3]                 ┆ [1, 2]             │
        └─────────┴─────────────────────┴────────────────────┘
    """
    _check_columns_in_df(df, [inview_col, clicked_col])
    negative_article_ids = (
        list(filter(lambda x: x not in clicked, inview))
        for inview, clicked in zip(df[inview_col].to_list(), df[clicked_col].to_list())
    )
    return df.with_columns(pl.Series(inview_col, list(negative_article_ids)))


# NEW FUNCTION
def flatten_single_value_list_column(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    """
    Flatten a column of lists to single values if each list contains only one element.

    Args:
        df (pl.DataFrame): The input Polars DataFrame.
        col_name (str): The name of the column to flatten.

    Returns:
        pl.DataFrame: A new DataFrame with the specified column flattened.
    """
    return df.with_columns(pl.col(col_name).arr.first().alias(col_name))


def sampling_strategy_wu2019(
        df: pl.DataFrame,
        npratio: int,
        shuffle: bool = False,
        with_replacement: bool = True,
        seed: int = None,
        inview_col: str = "article_id",
        clicked_col: str = "article_ids_clicked",
) -> pl.DataFrame:
    df = (
        # Step 1: Remove the positive 'article_id' from inview articles
        df.pipe(
            remove_positives_from_inview, inview_col=inview_col, clicked_col=clicked_col
        )
        # Step 2: Explode the DataFrame based on the clicked articles column
        .explode(clicked_col)
        # Step 3: Downsample the inview negative 'article_id' according to npratio (negative 'article_id' per positive 'article_id')
        .pipe(
            sample_article_ids,
            n=npratio,
            with_replacement=with_replacement,
            seed=seed,
            inview_col=inview_col,
        )
        # Step 4: Concatenate the clicked articles back to the inview articles as lists
        .with_columns(pl.concat_list([inview_col, clicked_col]))
        # Step 5: Convert clicked articles column to type List(Int):
        .with_columns(pl.col(inview_col).list.tail(1).alias(clicked_col))
    )
    if shuffle:
        df = shuffle_list_column(df, inview_col, seed)
    return df


def truncate_history(
        df: pl.DataFrame,
        column: str,
        history_size: int,
        padding_value: Any = None,
        enable_warning: bool = True,
) -> pl.DataFrame:
    """Truncates the history of a column containing a list of items.

    It is the tail of the values, i.e. the history ids should ascending order
    because each subsequent element (original timestamp) is greater than the previous element

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to truncate.
        history_size (int): The maximum size of the history to retain.
        padding_value (Any): Pad each list with specified value, ensuring
            equal length to each element. Default is None (no padding).
        enable_warning (bool): warn the user that history is expected in ascedings order.
            Default is True

    Returns:
        pl.DataFrame: A new DataFrame with the specified column truncated.

    Examples:
    >>> df = pl.DataFrame(
            {"id": [1, 2, 3], "history": [["a", "b", "c"], ["d", "e", "f", "g"], ["h", "i"]]}
        )
    >>> df
        shape: (3, 2)
        ┌─────┬───────────────────┐
        │ id  ┆ history           │
        │ --- ┆ ---               │
        │ i64 ┆ list[str]         │
        ╞═════╪═══════════════════╡
        │ 1   ┆ ["a", "b", "c"]   │
        │ 2   ┆ ["d", "e", … "g"] │
        │ 3   ┆ ["h", "i"]        │
        └─────┴───────────────────┘
    >>> truncate_history(df, 'history', 3)
        shape: (3, 2)
        ┌─────┬─────────────────┐
        │ id  ┆ history         │
        │ --- ┆ ---             │
        │ i64 ┆ list[str]       │
        ╞═════╪═════════════════╡
        │ 1   ┆ ["a", "b", "c"] │
        │ 2   ┆ ["e", "f", "g"] │
        │ 3   ┆ ["h", "i"]      │
        └─────┴─────────────────┘
    >>> truncate_history(df.lazy(), 'history', 3, '-').collect()
        shape: (3, 2)
        ┌─────┬─────────────────┐
        │ id  ┆ history         │
        │ --- ┆ ---             │
        │ i64 ┆ list[str]       │
        ╞═════╪═════════════════╡
        │ 1   ┆ ["a", "b", "c"] │
        │ 2   ┆ ["e", "f", "g"] │
        │ 3   ┆ ["-", "h", "i"] │
        └─────┴─────────────────┘
    """
    if enable_warning:
        function_name = inspect.currentframe().f_code.co_name
        warnings.warn(f"{function_name}: The history IDs expeced in ascending order")
    if padding_value is not None:
        df = df.with_columns(
            pl.col(column)
            .list.reverse()
            .list.eval(pl.element().extend_constant(padding_value, n=history_size))
            .list.reverse()
        )
    return df.with_columns(pl.col(column).list.tail(history_size))


def create_dynamic_history(
        df: pl.DataFrame,
        history_size: int,
        history_col: str = "history_dynamic",
        user_col: str = DEFAULT_USER_COL,
        item_col: str = DEFAULT_ARTICLE_ID_COL,
        timestamp_col: str = DEFAULT_IMPRESSION_TIMESTAMP_COL,
) -> pl.DataFrame:
    """Generates a dynamic history of user interactions with articles based on a given DataFrame.

    Beaware, the groupby_rolling will add all the Null values, which can only be removed afterwards.
    Unlike the 'create_fixed_history' where we first remove all the Nulls, we can only do this afterwards.
    As a results, the 'history_size' might be set to N but after removal of Nulls it is (N-n_nulls) long.

    Args:
        df (pl.DataFrame): A Polars DataFrame with columns 'user_id', 'article_id', and 'first_page_time'.
        history_size (int): The maximum number of previous interactions to include in the dynamic history for each user.

    Returns:
        pl.DataFrame: A new Polars DataFrame with the same columns as the input DataFrame, plus two new columns per user:
        - 'dynamic_article_id': a list of up to 'history_size' article IDs representing the user's previous interactions,
            ordered from most to least recent. If there are fewer than 'history_size' previous interactions, the list
            is padded with 'None' values.
    Raises:
        ValueError: If the input DataFrame does not contain columns 'user_id', 'article_id', and 'first_page_time'.

    Examples:
    >>> from RecSysChallenge2024_DIN.utils.constants import (
            DEFAULT_IMPRESSION_TIMESTAMP_COL,
            DEFAULT_ARTICLE_ID_COL,
            DEFAULT_USER_COL,
        )
    >>> df = pl.DataFrame(
            {
                DEFAULT_USER_COL: [0, 0, 0, 1, 1, 1, 0, 2],
                DEFAULT_ARTICLE_ID_COL: [
                    9604210,
                    9634540,
                    9640420,
                    9647983,
                    9647984,
                    9647981,
                    None,
                    None,
                ],
                DEFAULT_IMPRESSION_TIMESTAMP_COL: [
                    datetime.datetime(2023, 2, 18),
                    datetime.datetime(2023, 2, 18),
                    datetime.datetime(2023, 2, 25),
                    datetime.datetime(2023, 2, 22),
                    datetime.datetime(2023, 2, 21),
                    datetime.datetime(2023, 2, 23),
                    datetime.datetime(2023, 2, 19),
                    datetime.datetime(2023, 2, 26),
                ],
            }
        )
    >>> create_dynamic_history(df, 3)
        shape: (8, 4)
        ┌─────────┬────────────┬─────────────────────┬────────────────────┐
        │ user_id ┆ article_id ┆ impression_time     ┆ history_dynamic    │
        │ ---     ┆ ---        ┆ ---                 ┆ ---                │
        │ i64     ┆ i64        ┆ datetime[μs]        ┆ list[i64]          │
        ╞═════════╪════════════╪═════════════════════╪════════════════════╡
        │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ []                 │
        │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ [9604210]          │
        │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ [9604210, 9634540] │
        │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ [9604210, 9634540] │
        │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ []                 │
        │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ [9647984]          │
        │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ [9647984, 9647983] │
        │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ []                 │
        └─────────┴────────────┴─────────────────────┴────────────────────┘
    """
    _check_columns_in_df(df, [user_col, timestamp_col, item_col])
    GROUPBY_ID = generate_unique_name(df.columns, "_groupby_id")
    df = df.sort([user_col, timestamp_col])
    return (
        df.with_columns(
            # DYNAMIC HISTORY START
            df.with_row_index(name=GROUPBY_ID)
            .with_columns(pl.col([GROUPBY_ID]).cast(pl.Int64))
            .rolling(
                index_column=GROUPBY_ID,
                period=f"{history_size}i",
                closed="left",
                by=[user_col],
            )
            .agg(pl.col(item_col).alias(history_col))
            # DYNAMIC HISTORY END
        )
        .pipe(drop_nulls_from_list, column=history_col)
        .drop(GROUPBY_ID)
    )


def create_fixed_history(
        df: pl.DataFrame,
        dt_cutoff: datetime,
        history_size: int = None,
        history_col: str = "history_fixed",
        user_col: str = DEFAULT_USER_COL,
        item_col: str = DEFAULT_ARTICLE_ID_COL,
        timestamp_col: str = DEFAULT_IMPRESSION_TIMESTAMP_COL,
) -> pl.DataFrame:
    """
    Create fixed histories for each user in a dataframe of user browsing behavior.

    Args:
        df (pl.DataFrame): A dataframe with columns "user_id", "first_page_time", and "article_id", representing user browsing behavior.
        dt_cutoff (datetime): A datetime object representing the cutoff time. Only browsing behavior before this time will be considered.
        history_size (int, optional): The maximum number of previous interactions to include in the fixed history for each user (using tail). Default is None.
            If None, all interactions are included.

    Returns:
        pl.DataFrame: A modified dataframe with columns "user_id" and "fixed_article_id". Each row represents a user and their fixed browsing history,
        which is a list of article IDs. The "fixed_" prefix is added to distinguish the fixed history from the original "article_id" column.

    Raises:
        ValueError: If the input dataframe does not contain the required columns.

    Examples:
        >>> from RecSysChallenge2024_DIN.utils.constants import (
                DEFAULT_IMPRESSION_TIMESTAMP_COL,
                DEFAULT_ARTICLE_ID_COL,
                DEFAULT_USER_COL,
            )
        >>> df = pl.DataFrame(
                {
                    DEFAULT_USER_COL: [0, 0, 0, 1, 1, 1, 0, 2],
                    DEFAULT_ARTICLE_ID_COL: [
                        9604210,
                        9634540,
                        9640420,
                        9647983,
                        9647984,
                        9647981,
                        None,
                        None,
                    ],
                    DEFAULT_IMPRESSION_TIMESTAMP_COL: [
                        datetime.datetime(2023, 2, 18),
                        datetime.datetime(2023, 2, 18),
                        datetime.datetime(2023, 2, 25),
                        datetime.datetime(2023, 2, 22),
                        datetime.datetime(2023, 2, 21),
                        datetime.datetime(2023, 2, 23),
                        datetime.datetime(2023, 2, 19),
                        datetime.datetime(2023, 2, 26),
                    ],
                }
            )
        >>> dt_cutoff = datetime.datetime(2023, 2, 24)
        >>> create_fixed_history(df.lazy(), dt_cutoff).collect()
            shape: (8, 4)
            ┌─────────┬────────────┬─────────────────────┬─────────────────────────────┐
            │ user_id ┆ article_id ┆ impression_time     ┆ history_fixed               │
            │ ---     ┆ ---        ┆ ---                 ┆ ---                         │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ list[i64]                   │
            ╞═════════╪════════════╪═════════════════════╪═════════════════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ [9604210, 9634540]          │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ [9604210, 9634540]          │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ [9604210, 9634540]          │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ [9604210, 9634540]          │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ [9647984, 9647983, 9647981] │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ [9647984, 9647983, 9647981] │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ [9647984, 9647983, 9647981] │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ null                        │
            └─────────┴────────────┴─────────────────────┴─────────────────────────────┘
        >>> create_fixed_history(df.lazy(), dt_cutoff, 1).collect()
            shape: (8, 4)
            ┌─────────┬────────────┬─────────────────────┬───────────────┐
            │ user_id ┆ article_id ┆ impression_time     ┆ history_fixed │
            │ ---     ┆ ---        ┆ ---                 ┆ ---           │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ list[i64]     │
            ╞═════════╪════════════╪═════════════════════╪═══════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ [9634540]     │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ [9634540]     │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ [9634540]     │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ [9634540]     │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ [9647981]     │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ [9647981]     │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ [9647981]     │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ null          │
            └─────────┴────────────┴─────────────────────┴───────────────┘
    """
    _check_columns_in_df(df, [user_col, timestamp_col, item_col])

    df = df.sort(user_col, timestamp_col)
    df_history = (
        df.select(user_col, timestamp_col, item_col)
        .filter(pl.col(item_col).is_not_null())
        .filter(pl.col(timestamp_col) < dt_cutoff)
        .group_by(user_col)
        .agg(
            pl.col(item_col).alias(history_col),
        )
    )
    if history_size is not None:
        df_history = df_history.with_columns(
            pl.col(history_col).list.tail(history_size)
        )
    return df.join(df_history, on=user_col, how="left")


def create_fixed_history_aggr_columns(
        df: pl.DataFrame,
        dt_cutoff: datetime,
        history_size: int = None,
        columns: list[str] = [],
        suffix: str = "_fixed",
        user_col: str = DEFAULT_USER_COL,
        item_col: str = DEFAULT_ARTICLE_ID_COL,
        timestamp_col: str = DEFAULT_IMPRESSION_TIMESTAMP_COL,
) -> pl.DataFrame:
    """
    This function aggregates historical data in a Polars DataFrame based on a specified cutoff datetime and user-defined columns.
    The historical data is fixed to a given number of most recent records per user.

    Parameters:
        df (pl.DataFrame): The input Polars DataFrame OR LazyFrame.
        dt_cutoff (datetime): The cutoff datetime for filtering the history.
        history_size (int, optional): The number of most recent records to keep for each user.
            If None, all history before the cutoff is kept.
        columns (list[str], optional): List of column names to be included in the aggregation.
            These columns are in addition to the mandatory 'user_id', 'article_id', and 'impression_timestamp'.
        lazy_output (bool, optional): whether to output df as LazyFrame.

    Returns:
        pl.DataFrame: A new DataFrame with the original columns and added columns for each specified column in the history.
        Each new column contains a list of historical values.

    Raises:
        ValueError: If the input dataframe does not contain the required columns.

    Examples:
        >>> from RecSysChallenge2024_DIN.utils.constants import (
                DEFAULT_IMPRESSION_TIMESTAMP_COL,
                DEFAULT_ARTICLE_ID_COL,
                DEFAULT_READ_TIME_COL,
                DEFAULT_USER_COL,
            )
        >>> df = pl.DataFrame(
                {
                    DEFAULT_USER_COL: [0, 0, 0, 1, 1, 1, 0, 2],
                    DEFAULT_ARTICLE_ID_COL: [
                        9604210,
                        9634540,
                        9640420,
                        9647983,
                        9647984,
                        9647981,
                        None,
                        None,
                    ],
                    DEFAULT_IMPRESSION_TIMESTAMP_COL: [
                        datetime.datetime(2023, 2, 18),
                        datetime.datetime(2023, 2, 18),
                        datetime.datetime(2023, 2, 25),
                        datetime.datetime(2023, 2, 22),
                        datetime.datetime(2023, 2, 21),
                        datetime.datetime(2023, 2, 23),
                        datetime.datetime(2023, 2, 19),
                        datetime.datetime(2023, 2, 26),
                    ],
                    DEFAULT_READ_TIME_COL: [
                        0,
                        2,
                        8,
                        13,
                        1,
                        1,
                        6,
                        1
                    ],
                    "nothing": [
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                }
            )
        >>> dt_cutoff = datetime.datetime(2023, 2, 24)
        >>> columns = [DEFAULT_IMPRESSION_TIMESTAMP_COL, DEFAULT_READ_TIME_COL]
        >>> create_fixed_history_aggr_columns(df.lazy(), dt_cutoff, columns=columns).collect()
            shape: (8, 8)
            ┌─────────┬────────────┬─────────────────────┬───────────┬─────────┬─────────────────┬─────────────────────────────┬───────────────────────────────────┐
            │ user_id ┆ article_id ┆ impression_time     ┆ read_time ┆ nothing ┆ read_time_fixed ┆ article_id_fixed            ┆ impression_time_fixed             │
            │ ---     ┆ ---        ┆ ---                 ┆ ---       ┆ ---     ┆ ---             ┆ ---                         ┆ ---                               │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ i64       ┆ null    ┆ list[i64]       ┆ list[i64]                   ┆ list[datetime[μs]]                │
            ╞═════════╪════════════╪═════════════════════╪═══════════╪═════════╪═════════════════╪═════════════════════════════╪═══════════════════════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ 0         ┆ null    ┆ [0, 2]          ┆ [9604210, 9634540]          ┆ [2023-02-18 00:00:00, 2023-02-18… │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ 2         ┆ null    ┆ [0, 2]          ┆ [9604210, 9634540]          ┆ [2023-02-18 00:00:00, 2023-02-18… │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ 6         ┆ null    ┆ [0, 2]          ┆ [9604210, 9634540]          ┆ [2023-02-18 00:00:00, 2023-02-18… │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ 8         ┆ null    ┆ [0, 2]          ┆ [9604210, 9634540]          ┆ [2023-02-18 00:00:00, 2023-02-18… │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ 1         ┆ null    ┆ [1, 13, 1]      ┆ [9647984, 9647983, 9647981] ┆ [2023-02-21 00:00:00, 2023-02-22… │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ 13        ┆ null    ┆ [1, 13, 1]      ┆ [9647984, 9647983, 9647981] ┆ [2023-02-21 00:00:00, 2023-02-22… │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ 1         ┆ null    ┆ [1, 13, 1]      ┆ [9647984, 9647983, 9647981] ┆ [2023-02-21 00:00:00, 2023-02-22… │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ 1         ┆ null    ┆ null            ┆ null                        ┆ null                              │
            └─────────┴────────────┴─────────────────────┴───────────┴─────────┴─────────────────┴─────────────────────────────┴───────────────────────────────────┘
        >>> create_fixed_history_aggr_columns(df.lazy(), dt_cutoff, 1, columns=columns).collect()
            shape: (8, 8)
            ┌─────────┬────────────┬─────────────────────┬───────────┬─────────┬─────────────────┬──────────────────┬───────────────────────┐
            │ user_id ┆ article_id ┆ impression_time     ┆ read_time ┆ nothing ┆ read_time_fixed ┆ article_id_fixed ┆ impression_time_fixed │
            │ ---     ┆ ---        ┆ ---                 ┆ ---       ┆ ---     ┆ ---             ┆ ---              ┆ ---                   │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ i64       ┆ null    ┆ list[i64]       ┆ list[i64]        ┆ list[datetime[μs]]    │
            ╞═════════╪════════════╪═════════════════════╪═══════════╪═════════╪═════════════════╪══════════════════╪═══════════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ 0         ┆ null    ┆ [2]             ┆ [9634540]        ┆ [2023-02-18 00:00:00] │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ 2         ┆ null    ┆ [2]             ┆ [9634540]        ┆ [2023-02-18 00:00:00] │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ 6         ┆ null    ┆ [2]             ┆ [9634540]        ┆ [2023-02-18 00:00:00] │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ 8         ┆ null    ┆ [2]             ┆ [9634540]        ┆ [2023-02-18 00:00:00] │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ 1         ┆ null    ┆ [1]             ┆ [9647981]        ┆ [2023-02-23 00:00:00] │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ 13        ┆ null    ┆ [1]             ┆ [9647981]        ┆ [2023-02-23 00:00:00] │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ 1         ┆ null    ┆ [1]             ┆ [9647981]        ┆ [2023-02-23 00:00:00] │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ 1         ┆ null    ┆ null            ┆ null             ┆ null                  │
            └─────────┴────────────┴─────────────────────┴───────────┴─────────┴─────────────────┴──────────────────┴───────────────────────┘
        >>> create_fixed_history_aggr_columns(df.lazy(), dt_cutoff, 1).collect()
            shape: (8, 6)
            ┌─────────┬────────────┬─────────────────────┬───────────┬─────────┬──────────────────┐
            │ user_id ┆ article_id ┆ impression_time     ┆ read_time ┆ nothing ┆ article_id_fixed │
            │ ---     ┆ ---        ┆ ---                 ┆ ---       ┆ ---     ┆ ---              │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ i64       ┆ null    ┆ list[i64]        │
            ╞═════════╪════════════╪═════════════════════╪═══════════╪═════════╪══════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ 0         ┆ null    ┆ [9634540]        │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ 2         ┆ null    ┆ [9634540]        │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ 6         ┆ null    ┆ [9634540]        │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ 8         ┆ null    ┆ [9634540]        │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ 1         ┆ null    ┆ [9647981]        │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ 13        ┆ null    ┆ [9647981]        │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ 1         ┆ null    ┆ [9647981]        │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ 1         ┆ null    ┆ null             │
            └─────────┴────────────┴─────────────────────┴───────────┴─────────┴──────────────────┘
        >>> create_fixed_history_aggr_columns(df.lazy(), dt_cutoff, 1).head(1).collect()
            shape: (1, 6)
            ┌─────────┬────────────┬─────────────────────┬───────────┬─────────┬──────────────────┐
            │ user_id ┆ article_id ┆ impression_time     ┆ read_time ┆ nothing ┆ article_id_fixed │
            │ ---     ┆ ---        ┆ ---                 ┆ ---       ┆ ---     ┆ ---              │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ i64       ┆ null    ┆ list[i64]        │
            ╞═════════╪════════════╪═════════════════════╪═══════════╪═════════╪══════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ 0         ┆ null    ┆ [9634540]        │
            └─────────┴────────────┴─────────────────────┴───────────┴─────────┴──────────────────┘
    """
    _check_columns_in_df(df, [user_col, item_col, timestamp_col] + columns)
    aggr_columns = list(set([item_col] + columns))
    df = df.sort(user_col, timestamp_col)
    df_history = (
        df.select(pl.all())
        .filter(pl.col(item_col).is_not_null())
        .filter(pl.col(timestamp_col) < dt_cutoff)
        .group_by(user_col)
        .agg(
            pl.col(aggr_columns).suffix(suffix),
        )
    )
    if history_size is not None:
        for col in aggr_columns:
            df_history = df_history.with_columns(
                pl.col(col + suffix).list.tail(history_size)
            )
    return df.join(df_history, on="user_id", how="left")


def add_session_id_and_next_items(
        df: pl.DataFrame,
        session_length: datetime.timedelta = datetime.timedelta(minutes=30),
        shift_columns: list[str] = [],
        prefix: str = "next_",
        session_col: str = "session_id",
        user_col: str = DEFAULT_USER_COL,
        timestamp_col: str = DEFAULT_IMPRESSION_TIMESTAMP_COL,
        **kwargs,
):
    """
    Adding session IDs and shifting specified columns to create 'next_' features.

    This function processes a DataFrame to assign unique session IDs based on a specified session length and creates new columns by shifting existing columns.
    These new columns are intended to represent the 'next_' features in a session-based context, e.g. 'next_read_time'.

    Args:
        df (pl.DataFrame): The DataFrame to process.
        session_length (datetime.timedelta, optional): The length of a session, used to determine when a new session ID should be assigned.
            Defaults to 30 minutes.
        shift_columns (list[str], optional): A list of column names whose values will be shifted to create the 'next_' features.
            Defaults to an empty list. If empty, you will only enrich with 'session_id'.
        tqdm_disable (bool, optional): If True, disables the tqdm progress bar. Useful in environments where tqdm's output is undesirable.
            Defaults to False. This may take some time, might be worth seeing the progress.
        prefix (str, optional): The prefix to add to the shifted column names. Defaults to 'next_'.

    Returns:
        pl.DataFrame: A modified DataFrame with added session IDs and 'next_clicked' features.

    Examples:
        >>> from RecSysChallenge2024_DIN.utils.constants import (
                DEFAULT_IMPRESSION_TIMESTAMP_COL,
                DEFAULT_ARTICLE_ID_COL,
                DEFAULT_USER_COL,
            )
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame(
                {
                    DEFAULT_USER_COL: [1, 1, 2, 2],
                    DEFAULT_IMPRESSION_TIMESTAMP_COL: [
                        datetime.datetime(year=2023, month=1, day=1, minute=0),
                        datetime.datetime(year=2023, month=1, day=1, minute=20),
                        datetime.datetime(year=2023, month=1, day=1, minute=0),
                        datetime.datetime(year=2023, month=1, day=1, minute=35),
                    ],
                    DEFAULT_READ_TIME_COL: [9, 5, 1, 10],
                }
            )
        >>> add_session_id_and_next_items(df, datetime.timedelta(minutes=30), shift_columns=['read_time'])
            shape: (4, 5)
            ┌─────────┬─────────────────────┬───────────┬────────────┬────────────────┐
            │ user_id ┆ impression_time     ┆ read_time ┆ session_id ┆ next_read_time │
            │ ---     ┆ ---                 ┆ ---       ┆ ---        ┆ ---            │
            │ i64     ┆ datetime[μs]        ┆ i64       ┆ u32        ┆ i64            │
            ╞═════════╪═════════════════════╪═══════════╪════════════╪════════════════╡
            │ 1       ┆ 2023-01-01 00:00:00 ┆ 9         ┆ 0          ┆ 5              │
            │ 1       ┆ 2023-01-01 00:20:00 ┆ 5         ┆ 0          ┆ null           │
            │ 2       ┆ 2023-01-01 00:00:00 ┆ 1         ┆ 2          ┆ null           │
            │ 2       ┆ 2023-01-01 00:35:00 ┆ 10        ┆ 3          ┆ null           │
            └─────────┴─────────────────────┴───────────┴────────────┴────────────────┘
        >>> add_session_id_and_next_items(df, datetime.timedelta(minutes=60), shift_columns=['read_time'])
            shape: (4, 5)
            ┌─────────┬─────────────────────┬───────────┬────────────┬────────────────┐
            │ user_id ┆ impression_time     ┆ read_time ┆ session_id ┆ next_read_time │
            │ ---     ┆ ---                 ┆ ---       ┆ ---        ┆ ---            │
            │ i64     ┆ datetime[μs]        ┆ i64       ┆ u32        ┆ i64            │
            ╞═════════╪═════════════════════╪═══════════╪════════════╪════════════════╡
            │ 1       ┆ 2023-01-01 00:00:00 ┆ 9         ┆ 0          ┆ 5              │
            │ 1       ┆ 2023-01-01 00:20:00 ┆ 5         ┆ 0          ┆ null           │
            │ 2       ┆ 2023-01-01 00:00:00 ┆ 1         ┆ 2          ┆ 10             │
            │ 2       ┆ 2023-01-01 00:35:00 ┆ 10        ┆ 2          ┆ null           │
            └─────────┴─────────────────────┴───────────┴────────────┴────────────────┘
    """
    GROUPBY_ID = generate_unique_name(df.columns, "_groupby_id")
    # =>
    df = df.with_row_count(GROUPBY_ID)

    # => INCREMENTAL SESSION-ID:
    s_id = 0
    # => COLUMNS:
    next_shift_columns = [prefix + feat for feat in shift_columns]
    select_columns = list(set([user_col, timestamp_col, GROUPBY_ID] + shift_columns))
    # =>
    df_concat = []
    #
    _check_columns_in_df(df, select_columns)

    for df_user in tqdm(
            df.select(select_columns).partition_by(by=user_col),
            disable=kwargs.get("disable", False),
            ncols=kwargs.get("ncols", 80),
    ):
        df_session = (
            df_user.sort(timestamp_col)
            .groupby_dynamic(timestamp_col, every=session_length)
            .agg(
                GROUPBY_ID,
                pl.col(shift_columns).shift(-1).prefix(prefix),
            )
            .with_row_count(session_col, offset=s_id)
        )
        #
        s_id += df_user.shape[0]
        df_concat.append(df_session)

    df_concat = (
        pl.concat(df_concat)
        .lazy()
        .select(GROUPBY_ID, session_col, pl.col(next_shift_columns))
        .explode(GROUPBY_ID, pl.col(next_shift_columns))
        .collect()
    )
    return df.join(df_concat, on=GROUPBY_ID, how="left").drop(GROUPBY_ID)


def add_prediction_scores(
        df: pl.DataFrame,
        scores: Iterable[float],
        prediction_scores_col: str = "scores",
        inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
) -> pl.DataFrame:
    """
    Adds prediction scores to a DataFrame for the corresponding test predictions.

    Args:
        df (pl.DataFrame): The DataFrame to which the prediction scores will be added.
        test_prediction (Iterable[float]): A list, array or simialr of prediction scores for the test data.

    Returns:
        pl.DataFrame: The DataFrame with the prediction scores added.

    Raises:
        ValueError: If there is a mismatch in the lengths of the list columns.

    >>> from RecSysChallenge2024_DIN.utils.constants import DEFAULT_INVIEW_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "id": [1,2],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    [1, 2, 3],
                    [4, 5],
                ],
            }
        )
    >>> test_prediction = [[0.3], [0.4], [0.5], [0.6], [0.7]]
    >>> add_prediction_scores(df.lazy(), test_prediction).collect()
        shape: (2, 3)
        ┌─────┬─────────────┬────────────────────────┐
        │ id  ┆ article_ids ┆ prediction_scores_test │
        │ --- ┆ ---         ┆ ---                    │
        │ i64 ┆ list[i64]   ┆ list[f32]              │
        ╞═════╪═════════════╪════════════════════════╡
        │ 1   ┆ [1, 2, 3]   ┆ [0.3, 0.4, 0.5]        │
        │ 2   ┆ [4, 5]      ┆ [0.6, 0.7]             │
        └─────┴─────────────┴────────────────────────┘
    ## The input can can also be an np.array
    >>> add_prediction_scores(df.lazy(), np.array(test_prediction)).collect()
        shape: (2, 3)
        ┌─────┬─────────────┬────────────────────────┐
        │ id  ┆ article_ids ┆ prediction_scores_test │
        │ --- ┆ ---         ┆ ---                    │
        │ i64 ┆ list[i64]   ┆ list[f32]              │
        ╞═════╪═════════════╪════════════════════════╡
        │ 1   ┆ [1, 2, 3]   ┆ [0.3, 0.4, 0.5]        │
        │ 2   ┆ [4, 5]      ┆ [0.6, 0.7]             │
        └─────┴─────────────┴────────────────────────┘
    """
    GROUPBY_ID = generate_unique_name(df.columns, "_groupby_id")
    # df_preds = pl.DataFrame()
    scores = (
        df.lazy()
        .select(pl.col(inview_col))
        .with_row_index(GROUPBY_ID)
        .explode(inview_col)
        .with_columns(pl.Series(prediction_scores_col, scores).explode())
        .group_by(GROUPBY_ID)
        .agg(inview_col, prediction_scores_col)
        .sort(GROUPBY_ID)
        .collect()
    )
    return df.with_columns(scores.select(prediction_scores_col)).drop(GROUPBY_ID)


def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
    """
    Load ebnerd - function
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        # .pipe(
        #     #truncate_history,
        #     column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        #     history_size=history_size,
        #     padding_value=0,
        # )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes,
            df2=df_history.collect(),
            on=DEFAULT_USER_COL,
            how="left",
        )
    )
    return df_behaviors


def append_to_list(lst: list, news_df: pl.DataFrame, n_samples: int):
    """
    NEW!!!
    Function to extend a list with a number of article_ids taken from a news_df
    Args:
        lst:
        news_df:

    Returns:

    """
    lst = list(lst)
    samples = pl.Series(news_df.select("article_id").collect().sample(n=n_samples)).to_list()
    lst.extend(samples)
    return lst


def add_soft_neg_samples(df: pl.DataFrame, n_samples: int, news_df: pl.DataFrame):
    df = df.with_columns(pl.col("article_ids_inview").map_elements(lambda x: append_to_list(x, news_df, n_samples)))
    return df


def map_feat_id_func(df, column, seq_feat=False):
    feat_set = set(flatten(df[column].to_list()))
    map_dict = dict(zip(list(feat_set), range(1, 1 + len(feat_set))))
    if seq_feat:
        df = df.with_columns(pl.col(column).apply(lambda x: [map_dict.get(i, 0) for i in x]))
    else:
        df = df.with_columns(pl.col(column).apply(lambda x: map_dict.get(x, 0)).cast(str))
    return df


def tokenize_seq(df, column, map_feat_id=True, max_seq_length=5, sep="^"):
    df = df.with_columns(pl.col(column).apply(lambda x: x[-max_seq_length:]))
    if map_feat_id:
        df = map_feat_id_func(df, column, seq_feat=True)
    df = df.with_columns(pl.col(column).apply(lambda x: f"{sep}".join(str(i) for i in x)))
    return df


def impute_list_with_mean(lst):
    non_null_values = [x for x in lst if x not in [None, "null"]]
    if non_null_values:
        mean_value = sum(non_null_values) / len(non_null_values)
        return [x if x is not None else mean_value for x in lst]
    else:
        return lst


def encode_date_list(lst):
    return [x.timestamp() for x in lst]


def compute_near_realtime_ctr(behavior_df: pl.DataFrame, last_n_days=2) -> dict[str, float]:
    """Compute near realtime ctr for items based on their occurrence in user interactions of the last n days.

        This function calculates the near realtime ctr of each item as the fraction of users who have interacted with that item.
        The popularity score, p_i, for an item is defined as the number of interactions item received in the last n days divided by the
        total number of interactions in the last n days.

        Note:
            Each entry can only have the same item ones.

        Args:
            R (Iterable[np.ndarray]): An iterable of numpy arrays, where each array represents the items interacted with by a single user.
                Each element in the array should be a string identifier for an item.

        Returns:
            dict[str, float]: A dictionary where keys are item identifiers and values are their corresponding near realtime ctr (as floats).
    """
    # Collect necessary columns from the DataFrames
    impression_time_threshold = list(behavior_df['impression_time'].sort(descending=True))[0].date() - timedelta(
        days=last_n_days - 1)
    behavior_df = behavior_df.filter(pl.col('impression_time') >= impression_time_threshold).select(
        ['user_id', 'article_ids_clicked'])
    # Explode the lists to have one article ID per row
    behavior_df = behavior_df.explode('article_ids_clicked')
    # Group by user_id and aggregate the article IDs into a list
    behavior_df = behavior_df.groupby('user_id').agg(pl.col('article_ids_clicked'))
    # Convert to list of np.array
    last_n_days_clicked_list = [np.unique(np.array(ids)) for ids in behavior_df['article_ids_clicked'].to_list()]

    R_flatten = np.concatenate(last_n_days_clicked_list)
    item_counts = Counter(R_flatten)
    return {item: (r_ui / len(R_flatten)) for item, r_ui in item_counts.items()}


def get_enriched_user_history(behavior_df: pl.DataFrame, history_df: pl.DataFrame) -> list[
    np.array]:
    behavior_df = behavior_df.select(['user_id', 'article_ids_clicked'])
    history_df = history_df.select(['user_id', 'article_id_fixed'])

    # Explode the lists to have one article ID per row
    behavior_df = behavior_df.explode('article_ids_clicked')
    history_df = history_df.explode('article_id_fixed')

    # Rename columns to match before concatenation
    behavior_df = behavior_df.rename({'article_ids_clicked': 'article_id'})
    history_df = history_df.rename({'article_id_fixed': 'article_id'})

    # Combine the behavior and history DataFrames
    combined_df = pl.concat([history_df, behavior_df])

    # Group by user_id and aggregate the article IDs into a list
    enriched_history = combined_df.groupby('user_id').agg(pl.col('article_id').alias('article_ids'))

    # Convert to list of np.array
    enriched_history_list = [np.unique(np.array(ids)) for ids in enriched_history['article_ids'].to_list()]

    return enriched_history_list


def compute_item_interactions(history_df: pl.DataFrame):
    history_df = history_df.select(['user_id', 'article_id_fixed'])
    # Convert to list of np.array
    R = [np.unique(np.array(ids)) for ids in history_df['article_id_fixed'].to_list()]
    R_flatten = np.concatenate(R)
    item_counts = Counter(R_flatten)
    return {item: r_ui for item, r_ui in item_counts.items()}


def compute_user_interactions(history_df: pl.DataFrame):
    history_df = history_df.select(['user_id', 'article_id_fixed'])
    # Convert to list of np.array
    return {user_id: len(np.unique(np.array(ids))) for user_id, ids in history_df.iter_rows()}


def k_core(history_df: pl.DataFrame, k=5):
    print(f"\nInitial number of users: {len(history_df)}")
    print(
        f"Initial number of items: {len([el for ids in history_df['article_id_fixed'].to_list() for el in np.unique(np.array(ids))])}")
    user_counts = compute_user_interactions(history_df)
    # filter user by history length
    user_counts = {user_id: history_len for user_id, history_len in user_counts.items() if history_len >= k}
    history_df = history_df.filter(pl.col('user_id').is_in(list(user_counts.keys())))
    # filter items by number of interactions
    item_counts = compute_item_interactions(history_df)
    item_counts = {item_id: num_interactions for item_id, num_interactions in item_counts.items() if
                   num_interactions >= k}
    history_df = history_df.with_columns(
        pl.col('article_id_fixed').map_elements(lambda x: remove_elements_from_lst(x, item_counts.keys())))
    print(f"Final number of users: {len(history_df)}")
    print(
        f"Final number of items: {len([el for ids in history_df['article_id_fixed'].to_list() for el in np.unique(np.array(ids))])}")
    return history_df


def iterative_k_core(history_df: pl.DataFrame, k=5):
    condition = True
    while condition:
        num_usrs = len(history_df)
        num_items = len([el for ids in history_df['article_id_fixed'].to_list() for el in np.unique(np.array(ids))])

        history_df = k_core(history_df, k)
        condition = not ((num_usrs == len(history_df)) and (num_items == len(
            [el for ids in history_df['article_id_fixed'].to_list() for el in np.unique(np.array(ids))])))

    return history_df


def remove_elements_from_lst(lst, elements):
    return [x for x in lst if x in elements]


def compute_item_popularity_scores(R: Iterable[np.ndarray]) -> dict[str, float]:
    """Compute popularity scores for items based on their occurrence in user interactions.

    This function calculates the popularity score of each item as the fraction of users who have interacted with that item.
    The popularity score, p_i, for an item is defined as the number of users who have interacted with the item divided by the
    total number of users.

    Formula:
        p_i = | {u ∈ U}, r_ui != Ø | / |U|

    where p_i is the popularity score of an item, U is the total number of users, and r_ui is the interaction of user u with item i (non-zero
    interaction implies the user has seen the item).

    Note:
        Each entry can only have the same item ones.

    Args:
        R (Iterable[np.ndarray]): An iterable of numpy arrays, where each array represents the items interacted with by a single user.
            Each element in the array should be a string identifier for an item.

    Returns:
        dict[str, float]: A dictionary where keys are item identifiers and values are their corresponding popularity scores (as floats).

    Examples:
    >>> R = [
            np.array(["item1", "item2", "item3"]),
            np.array(["item1", "item3"]),
            np.array(["item1", "item4"]),
        ]
    >>> print(compute_item_popularity_scores(R))
        {'item1': 1.0, 'item2': 0.3333333333333333, 'item3': 0.6666666666666666, 'item4': 0.3333333333333333}
    """
    U = len(R)
    R_flatten = np.concatenate(R)
    item_counts = Counter(R_flatten)
    return {item: (r_ui / U) for item, r_ui in item_counts.items()}


def create_inviews_vectors(behavior_df, emb_df):
    # Explode the article_ids_inview column to have one article_id per row
    exploded_behavior_df = behavior_df.select('impression_id', 'article_ids_inview', ).explode('article_ids_inview')

    del behavior_df
    gc.collect()
    # Rename the exploded column for joining
    exploded_behavior_df = exploded_behavior_df.rename({'article_ids_inview': 'article_id'})

    # Perform a join to get the contrastive vectors for all article_ids
    joined_df = exploded_behavior_df.join(emb_df, on='article_id', how='left')

    del exploded_behavior_df
    gc.collect()

    # Group by impression_id and aggregate the vectors to create the mean vector for each impression_id
    inviews_vectors_df = (
        joined_df
        .groupby('impression_id')
        .agg(
            pl.col(emb_df.columns[-1]).apply(lambda x: np.array(x).mean(axis=0).tolist()).alias(
                'inview_vector_mean')
        )
    )
    del joined_df
    gc.collect()

    impression_ids = inviews_vectors_df['impression_id']
    inviews_vectors = np.vstack(inviews_vectors_df['inview_vector_mean'].to_list())

    return impression_ids, inviews_vectors


def clean_dataframe(row):
    return (row[0], list(set([x for xs in row[1] for x in xs])))


def exponential_decay(freshness, alpha=0.1):
    return np.exp(-alpha * freshness)
