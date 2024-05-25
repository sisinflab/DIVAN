import os
import zipfile
import requests
import io


def download_ebnerd_dataset(dataset_size, train_path, val_path, test_path=None):
    dataset_url = f"https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_{dataset_size}.zip"
    test_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip"
    contrast_emb_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip"
    image_emb_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_image_embeddings.zip"

    print(f"Getting Ebnerd {dataset_size} from : {dataset_url} ..")
    response_dataset = requests.get(dataset_url)
    with zipfile.ZipFile(io.BytesIO(response_dataset.content)) as zip_ref:
        zip_ref.extract("articles.parquet", path=train_path)
        zip_ref.extract("train/history.parquet")
        zip_ref.extract("train/behaviors.parquet")

        zip_ref.extract("validation/history.parquet")
        zip_ref.extract("validation/behaviors.parquet")

    if test_path:
        print(f"Getting Ebnerd test from : {test_url} ..")

        response_test = requests.get(test_url)

        with zipfile.ZipFile(io.BytesIO(response_test.content)) as zip_ref:
            zip_ref.extract("ebnerd_testset/articles.parquet")
            zip_ref.extract("ebnerd_testset/test/history.parquet")
            zip_ref.extract("ebnerd_testset/test/behaviors.parquet")
        os.rename("./ebnerd_testset/test", "./test")
        os.rename("./ebnerd_testset/articles.parquet", os.path.join(test_path, "articles.parquet"))
        os.removedirs("./ebnerd_testset")

    print(f"Getting news image embeddings from : {image_emb_url} ..")
    response_image_emb = requests.get(image_emb_url)
    with zipfile.ZipFile(io.BytesIO(response_image_emb.content)) as zip_ref:
        zip_ref.extract("Ekstra_Bladet_image_embeddings/image_embeddings.parquet")
    os.rename("./Ekstra_Bladet_image_embeddings/image_embeddings.parquet", "image_embeddings.parquet")
    os.removedirs("./Ekstra_Bladet_image_embeddings")

    print(f"Getting news contrastive embeddings from : {contrast_emb_url} ..")
    response_contrast_emb = requests.get(contrast_emb_url)
    with zipfile.ZipFile(io.BytesIO(response_contrast_emb.content)) as zip_ref:
        zip_ref.extract("Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet")
    os.rename("Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet", "contrastive_vector.parquet")
    os.removedirs("./Ekstra_Bladet_contrastive_vector")

    print(f"Dataset Ebnerd {dataset_size} downloaded!")
