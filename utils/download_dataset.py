import zipfile
import requests
import io


def download_ebnerd_dataset(dataset_size, train_path, test_path=None):
    dataset_url = f"https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_{dataset_size}.zip"
    test_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip"
    contrast_emb_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip"
    image_emb_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_image_embeddings.zip"

    print(f"Getting Ebnerd {dataset_size} from : {dataset_url} ..")
    response_dataset = requests.get(dataset_url)

    with zipfile.ZipFile(io.BytesIO(response_dataset.content)) as zip_ref:
        zip_ref.extract("articles.parquet", path=train_path)
    with zipfile.ZipFile(io.BytesIO(response_dataset.content)) as zip_ref:
        zip_ref.extract("train/history.parquet")
    with zipfile.ZipFile(io.BytesIO(response_dataset.content)) as zip_ref:
        zip_ref.extract("train/behaviors.parquet")
    with zipfile.ZipFile(io.BytesIO(response_dataset.content)) as zip_ref:
        zip_ref.extract("validation/history.parquet")
    with zipfile.ZipFile(io.BytesIO(response_dataset.content)) as zip_ref:
        zip_ref.extract("validation/behaviors.parquet")

    if test_path:
        print(f"Getting Ebnerd test from : {test_url} ..")
        response_test = requests.get(test_url)

        with zipfile.ZipFile(io.BytesIO(response_test.content)) as zip_ref:
            zip_ref.extract("articles.parquet", path=test_path)
        with zipfile.ZipFile(io.BytesIO(response_test.content)) as zip_ref:
            zip_ref.extract("test/history.parquet")
        with zipfile.ZipFile(io.BytesIO(response_test.content)) as zip_ref:
            zip_ref.extract("test/behaviors.parquet")

    print(f"Getting news image embeddings from : {image_emb_url} ..")
    response_image_emb = requests.get(image_emb_url)
    with zipfile.ZipFile(io.BytesIO(response_image_emb.content)) as zip_ref:
        zip_ref.extract("image_embeddings.parquet")

    print(f"Getting news contrastive embeddings from : {contrast_emb_url} ..")
    response_contrast_emb = requests.get(contrast_emb_url)
    with zipfile.ZipFile(io.BytesIO(response_contrast_emb.content)) as zip_ref:
        zip_ref.extract("contrastive_vector.parquet")

    print("All done!")
