import os
import zipfile
import requests
from tqdm import tqdm


def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data_chunk in response.iter_content(block_size):
            progress_bar.update(len(data_chunk))
            file.write(data_chunk)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


def extract_file(zip_filename, file_to_extract, destination_path):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extract(file_to_extract, path=destination_path)


def download_ebnerd_dataset(dataset_size, train_path, test_path=None):
    dataset_url = f"https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_{dataset_size}.zip"
    test_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip"
    contrast_emb_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip"
    image_emb_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_image_embeddings.zip"

    print(f"Getting Ebnerd {dataset_size} from : {dataset_url} ..")
    download_file(dataset_url, 'dataset.zip')
    extract_file('dataset.zip', "articles.parquet", train_path)
    extract_file('dataset.zip', "train/history.parquet", train_path)
    extract_file('dataset.zip', "train/behaviors.parquet", train_path)
    extract_file('dataset.zip', "validation/history.parquet", train_path)
    extract_file('dataset.zip', "validation/behaviors.parquet", train_path)
    os.remove("dataset.zip")

    if test_path:
        print(f"Getting Ebnerd test from : {test_url} ..")
        download_file(test_url, 'test.zip')
        extract_file('test.zip', "ebnerd_testset/articles.parquet", test_path)
        extract_file('test.zip', "ebnerd_testset/test/history.parquet", test_path)
        extract_file('test.zip', "ebnerd_testset/test/behaviors.parquet", test_path)
        os.remove("test.zip")

    print(f"Getting news image embeddings from : {image_emb_url} ..")
    download_file(image_emb_url, 'image_embeddings.zip')
    extract_file('image_embeddings.zip', "Ekstra_Bladet_image_embeddings/image_embeddings.parquet", "./")
    os.remove("image_embeddings.zip")

    print(f"Getting news contrastive embeddings from : {contrast_emb_url} ..")
    download_file(contrast_emb_url, 'contrastive_vector.zip')
    extract_file('contrastive_vector.zip', "Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet", "./")
    os.remove("contrastive_vector.zip")

    print("All done!")
