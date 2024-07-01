import logging
import os
import polars as pl
import shutil


def grank(x):
    scores = x["score"].tolist()
    tmp = [(i, s) for i, s in enumerate(scores)]
    tmp = sorted(tmp, key=lambda y: y[-1], reverse=True)
    rank = [(i + 1, t[0]) for i, t in enumerate(tmp)]
    rank = [str(r[0]) for r in sorted(rank, key=lambda y: y[-1])]
    rank = "[" + ",".join(rank) + "]"
    return str(x["impression_id"].iloc[0]) + " " + rank


experiment_id = "popular_ranker"
dataset = "large"
print("Reading test set...")
ans = pl.scan_csv(f"./data/Ebnerd_{dataset}/Ebnerd_{dataset}_pop/test.csv")
ans = ans.select(['impression_id', 'user_id', 'popularity_score'])
logging.info("Predicting scores...")
ans = ans.rename({'popularity_score': 'score'}).collect().to_pandas()
logging.info("Ranking samples...")
ans = ans.groupby(['impression_id', 'user_id'], sort=False).apply(grank).reset_index(drop=True)
logging.info("Writing results...")
os.makedirs("submit", exist_ok=True)
with open('submit/predictions.txt', "w") as fout:
    fout.write("\n".join(ans.to_list()))
shutil.make_archive(f'submit/{experiment_id}', 'zip', 'submit/', 'predictions.txt')
logging.info("All done.")
