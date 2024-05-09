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


def download_ebnerd_dataset(dataset_size, train_path, val_path, test_path=None):
    dataset_url = f"https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_{dataset_size}.zip"
    test_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip"
    contrast_emb_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip"
    image_emb_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_image_embeddings.zip"

    print(f"Getting Ebnerd {dataset_size} from : {dataset_url} ..")
    download_file(dataset_url, "dataset.zip")
    with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
        zip_ref.extract("articles.parquet", path=train_path)
        zip_ref.extract("train/history.parquet", path=train_path)
        zip_ref.extract("train/behaviors.parquet", path=train_path)

        zip_ref.extract("validation/history.parquet", path=val_path)
        zip_ref.extract("validation/behaviors.parquet", path=val_path)

    os.remove("dataset.zip")

    if test_path:
        print(f"Getting Ebnerd test from : {test_url} ..")
        download_file(test_url, 'test.zip')

        with zipfile.ZipFile("test.zip", "r") as zip_ref:
            zip_ref.extract("ebnerd_testset/articles.parquet")
            zip_ref.extract("ebnerd_testset/test/history.parquet")
            zip_ref.extract("ebnerd_testset/test/behaviors.parquet")

            os.rename("ebnerd_testset/articles.parquet", os.path.join(test_path, "articles.parquet"))
            os.rename("ebnerd_testset/test/history.parquet", os.path.join(test_path, "history.parquet"))
            os.rename("ebnerd_testset/test/behaviors.parquet", os.path.join(test_path, "behaviors.parquet"))

        os.removedirs("ebnerd_testset")
        os.remove("test.zip")

    print(f"Getting news image embeddings from : {image_emb_url} ..")
    download_file(image_emb_url, 'image_embeddings.zip')
    with zipfile.ZipFile("image_embeddings.zip", "r") as zip_ref:
        zip_ref.extract("Ekstra_Bladet_image_embeddings/image_embeddings.parquet")
        os.rename("Ekstra_Bladet_image_embeddings/image_embeddings.parquet", "./")
    os.removedirs("Ekstra_Bladet_image_embeddings")
    os.remove("image_embeddings.zip")

    print(f"Getting news contrastive embeddings from : {contrast_emb_url} ..")
    download_file(contrast_emb_url, 'contrastive_vector.zip')
    with zipfile.ZipFile("contrastive_vector.zip", "r") as zip_ref:
        zip_ref.extract("Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet")
        os.rename("Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet", "./")
    os.removedirs("Ekstra_Bladet_contrastive_vector")
    os.remove("contrastive_vector.zip")

    print("All done!")
