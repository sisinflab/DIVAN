## RecSysChallenge2024_DIN

### Environment

Please set up the environment as follows.
```
conda create -n recsys_din python==3.9
pip install -r requirements.txt
source activate recsys_din
```

### Data Preparation

1. Download the datasets at: https://recsys.eb.dk/#dataset

2. Unzip the data files to the following

    ```bash
    cd ~/RecSys2024_CTR_Challenge/data/Ebnerd_large/
    find -L .

    .
    ./train
    ./train/history.parquet
    ./train/articles.parquet
    ./train/behaviors.parquet
    ./validation
    ./validation/history.parquet
    ./validation/behaviors.parquet
    ./test
    ./test/history.parquet
    ./test/articles.parquet
    ./test/behaviors.parquet
    ./image_embeddings.parquet
    ./contrastive_vector.parquet
    ./prepare_data_v1.py
    ```

3. Convert the data to csv format

    ```bash
    cd ~/RecSys2024_CTR_Challenge/data/Ebnerd_large/
    python prepare_data_v1.py
    ```

### Version 1

1. Train the model on train and validation sets:

    ```
    python run_param_tuner.py --config config/DIN_ebnerd_large_x1_tuner_config_01.yaml --gpu 0
    ```

2. Make predictions on the test set:

    Get the experiment_id from running logs or the result csv file, and then you can run prediction on the test.

    ```
    python submit.py --config config/DIN_ebnerd_large_x1_tuner_config_01 --expid DIN_ebnerd_large_x1_001_1860e41e --gpu 1
    ```