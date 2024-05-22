from collections import Counter
from typing import Iterable
import numpy as np
from pandas.core.common import flatten
import polars as pl
import os

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


def get_enriched_user_history(behavior_df: pl.LazyFrame, history_df: pl.LazyFrame) -> list[np.array]:
    # Collect necessary columns from the DataFrames
    behavior_df = behavior_df.select(['user_id', 'article_ids_clicked']).collect()
    history_df = history_df.select(['user_id', 'article_id_fixed']).collect()

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
    enriched_history_list = [np.array(ids) for ids in enriched_history['article_ids'].to_list()]

    return enriched_history_list


def compute_item_popularity_scores(R: Iterable[np.array]) -> dict[str, float]:
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
