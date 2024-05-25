## RecSysChallenge2024_DIN

### If you want to use venv

1. Please set up the environment as follows.
   ```
   sudo apt-get install python3.9
   python3.9 -m venv recsys_din
   source recsys_din/bin/activate
   python -m pip install --upgrade pip
   pip install --no-cache-dir -r requirements.txt
   ```

## If you want to use Docker
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

### Data Preparation

1. Download and preprare data

    ```bash
    cd ./data/Ebnerd_large/
    python prepare_data_v1.py
    ```

2. Train the model on train and validation sets:

    ```bash
    python run_param_tuner.py --config config/DIN_ebnerd_large_x1_tuner_config_01.yaml --gpu 0
    ```

3. Make predictions on the test set:

    Get the experiment_id from running logs or the result csv file DIN_ebnerd_large_x1_tuner_config_01.csv, and then you can run prediction on the test.

    ```bash
    python submit.py --config config/DIN_ebnerd_large_x1_tuner_config_01 --expid DIN_ebnerd_large_x1_001_1860e41e --gpu 1
    ```