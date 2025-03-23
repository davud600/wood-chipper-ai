import requests
import os

from utils import DOWNLOADS_DIR


def download_s3_file(signed_get_url: str, output_filename: str):
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)

    file_path = os.path.join(DOWNLOADS_DIR, output_filename)

    response = requests.get(signed_get_url)
    if not response.ok:
        raise Exception("Failed to fetch document from S3.")

    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"File saved to {file_path}")


def upload_file_to_s3(signed_put_url: str, path: str):
    content_type = "application/pdf"

    with open(path, "rb") as f:
        file_data = f.read()

    response = requests.put(
        signed_put_url,
        data=file_data,
        headers={
            "Content-Type": content_type,
        },
    )

    if not response.ok:
        raise Exception(
            f"Failed to upload file to S3. Status: {response.status_code}, Body: {response.text}"
        )

    print(f"File uploaded successfully to S3.")
