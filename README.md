## RecSysChallenge2024_DIN

This repository is built on top of [FuxiCTR](https://github.com/reczoo/FuxiCTR), a configurable, tunable, and reproducible library for CTR prediction.

Paper reference FuxiCTR:
+ Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, Xiuqiang He. [Open Benchmarking for Click-Through Rate Prediction](https://arxiv.org/abs/2009.05794). *The 30th ACM International Conference on Information and Knowledge Management (CIKM)*, 2021.

## Setup virtual environment
### If you want to use venv

1. Please set up the environment as follows (we used python 3.9 and python 3.10).
   ```
   python3 -m venv recsys_din
   source recsys_din/bin/activate
   python -m pip install --upgrade pip
   pip install --no-cache-dir -r requirements.txt
   ```

### If you want to use Docker
1. make sure you have started the docker engine
2. Build the container:
   ```bash
      docker build -t recsyschallenge2024_din .
   ```
3. Run the container
   ```bash
      docker run -d -it --name recsyschallenge_container  recsyschallenge2024_din /bin/bash
   ```
4. Access the terminal of the container
   ```bash
   docker exec -it recsyschallenge_container /bin/sh
   ```

## Data Preparation and Model Training (DIVAN and VDIN)

1. Download and preprare data

    ```bash
    python prepare_data_v1.py --size large --test --embedding_size 64 --neg_sampling
    ```

2. Train the model on train and validation sets:
   - Train VDIN
       ```bash
       python run_param_tuner.py --config config/VDIN_ebnerd_large_x1_tuner_config_01.yaml --gpu 0
       ```
   - Train DIVAN
       ```bash
       python run_param_tuner.py --config config/DIVAN_ebnerd_large_x1_tuner_config_01.yaml --gpu 0
       ```

3. Make predictions on the test set:

    Get the experiment_id from running logs or the result csv file [VDIN|DIVAN]_ebnerd_large_x1_tuner_config_01.csv, and then you can run prediction on the test.

    ```bash
    python submit.py --config config/[VDIN|DIVAN]_ebnerd_large_x1_tuner_config_01 --expid [VDIN|DIVAN]_ebnerd_large_x1_001_1860e41e --gpu 0
    ```
   
## Data preparation and prediction with PopularRanker and ViralRanker
1. Download and preprare data

    ```bash
    python prepare_data_pop_and_vir_scores.py --size large --test
    ```
2. Test the model on the validation set:
    ```bash
    python run_[popular|virality]_expid.py
    ```
3. Make predictions on the test set:
   ```bash
    python submit_[popular|viral].py
    ```