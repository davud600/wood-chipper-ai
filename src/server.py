from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from transformers import AutoTokenizer

import redis
import torch
import fitz

# import csv  # temp: debugging.

from src.utils import MODEL_PATH, pages_to_append
from src.custom_types import FileContents
from src.model.model import SplitterModel
from src.api.s3 import download_s3_file, upload_file_to_s3
from src.api.document_records import (
    add_document_to_client_queue,
    create_document_record,
    notify_for_finished_splitting,
)
from src.document_processor.index import (
    convert_pdf_page_to_image,
    create_sub_document,
    get_image_contents,
    is_first_page,
)


# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
MAX_WORKERS = 1

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
r = redis.Redis(host="redis", port=6379, decode_responses=True)
contents_dict: FileContents = {}


# --------------------------------------------------------
# LOAD MODEL & TOKENIZER
# --------------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", device="cuda")

print("Loading model...")
model = SplitterModel().to("cuda")
model.eval()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=False, map_location="cuda"))


# --------------------------------------------------------
# ENDPOINT - SPLIT
# --------------------------------------------------------
@app.route("/split", methods=["POST"])
def process_split_endpoint():
    """
    Receives a POST request with "transaction_id", "document_id" and "signed_get_url" in body.

    documentId -> id of document record in database.
    seignedGetUrl -> aws s3 bucket signed url for downloading the file.

    Offloads the heavy lifting to a thread in ThreadPoolExecutor.
    Responds with 200 status code, actual response is sent via web socket event.
    """

    print("received request to split document...")
    data = request.get_json() or {}

    if "token" not in data:
        return jsonify({"error": "Missing 'token' in request body"}), 400
    if "transaction_id" not in data:
        return jsonify({"error": "Missing 'transaction_id' in request body"}), 400
    if "document_id" not in data:
        return jsonify({"error": "Missing 'document_id' in request body"}), 400
    if "signed_get_url" not in data:
        return jsonify({"error": "Missing 'signed_get_url' in request body"}), 400

    token = data.get("token")
    transaction_id = data.get("transaction_id")
    document_id = data.get("document_id")
    signed_get_url = data.get("signed_get_url")

    if not isinstance(token, str):
        return jsonify({"error": "'token' must be a string"}), 400
    if not isinstance(transaction_id, str) and not isinstance(transaction_id, int):
        return jsonify({"error": "'transaction_id' must be a number"}), 400
    if not isinstance(document_id, str) and not isinstance(document_id, int):
        return jsonify({"error": "'document_id' must be a number"}), 400
    if not isinstance(signed_get_url, str):
        return jsonify({"error": "'signed_get_url' must be a string"}), 400

    executor.submit(
        process_split_request,
        token,
        int(transaction_id),
        int(document_id),
        signed_get_url,
    )

    return jsonify({"message": "Processing started"}), 202


def process_split_request(
    token: str,
    transaction_id: int,
    parent_document_id: int,
    signed_get_url: str,
):
    """
    The actual worker function that handles the request data.
    (Runs in a separate thread from the main Flask thread.)
    """

    merged_file_name = f"{parent_document_id}.pdf"
    print(f"downloading source file {merged_file_name}...")
    merged_file_path = download_s3_file(signed_get_url, merged_file_name)
    merged_doc = fitz.open(merged_file_path)

    prev_split_page = 0

    for i in range(0, len(merged_doc), 1):
        # extract page contents if not on dict.
        page_content = contents_dict.get(i)
        if page_content is None:
            page_image = convert_pdf_page_to_image(merged_file_name, i)
            page_content = get_image_contents(page_image)
            r.set(f"{merged_file_name}-{i}", page_content)

        # extract contents of next pages if not on dict.
        content = f"<curr_page>{page_content}</curr_page>"
        for j in range(1, min(pages_to_append, len(merged_doc) - i), 1):
            next_page_content = r.get(f"{merged_file_name}-{i + j}")
            if next_page_content is None:
                next_page_image = convert_pdf_page_to_image(merged_file_name, i + j)
                next_page_content = get_image_contents(next_page_image)
                r.set(f"{merged_file_name}-{i + j}", next_page_content)

            content += f"<next_page_{j}>{next_page_content}</next_page_{j}>"

        print(f"page {i + 1} content:", content[:100] + "...")

        if i == 0:
            continue

        if is_first_page(tokenizer, model, content):
            # bg task...
            print(f"found first page: {i + 1}")

            try:
                # fetch web server to create doc rec, and get doc id & signed put url.
                print("creating document record...")
                document_record_id, signed_put_url = create_document_record(
                    token, transaction_id, parent_document_id
                )

                # create sub doc file.
                print(f"creating sub document {prev_split_page} - {i}...")
                sub_document_path = create_sub_document(
                    merged_file_name, prev_split_page, i, document_record_id
                )

                # save file to s3.
                print("uploading file to s3...")
                upload_file_to_s3(signed_put_url, sub_document_path)

                # send request to web server when upload is finished.
                print("notifying web server to add sub document to client queue...")
                add_document_to_client_queue(token, document_record_id)
                prev_split_page = i
            except Exception as e:
                print(e)

    # notify web server that splitting is finished.
    notify_for_finished_splitting(token, parent_document_id)


if __name__ == "__main__":
    print("Starting server...")
    app.run(host="0.0.0.0", port=8000, threaded=True)
