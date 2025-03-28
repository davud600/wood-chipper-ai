from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from typing import cast

import threading
import redis
import torch
import fitz

# import csv  # temp: debugging.

from src.utils import SPLITTER_MODEL_PATH, pages_to_append
from src.model.model import SplitterModel
from src.api.s3 import download_s3_file, upload_file_to_s3
from src.api.document_records import (
    add_document_to_client_queue,
    create_document_record,
    notify_for_finished_processing,
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
r = redis.Redis(host="localhost", port=6379, decode_responses=True)


# --------------------------------------------------------
# LOAD MODEL & TOKENIZER
# --------------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", device="cuda")

print("Loading model...")
model = SplitterModel().to("cuda")
model.eval()
model.load_state_dict(
    torch.load(SPLITTER_MODEL_PATH, weights_only=False, map_location="cuda")
)


# --------------------------------------------------------
# MAIN FUNCTIONS.
# --------------------------------------------------------
def handle_first_page(
    page: int,
    prev_split_page: int,
    token: str,
    transaction_id: int,
    parent_document_id: int,
    merged_file_name: str,
):
    """
    page -> 0-based, model predicted this as start of new document.
    prev_split_page -> previous first page.
    token -> client auth token.
    transaction_id -> id of transaction associated with parent document.
    parent_document_id -> merged document record id.
    merged_file_name -> file name of merged document.
    """

    print("creating document record...")
    document_record_id, signed_put_url = create_document_record(
        token, transaction_id, parent_document_id
    )

    print(f"creating sub document {prev_split_page} - {page}...")
    sub_document_path = create_sub_document(
        merged_file_name, prev_split_page, page, document_record_id
    )

    print("uploading file to s3...")
    upload_file_to_s3(signed_put_url, sub_document_path)

    print("notifying web server to add sub document to client queue...")
    add_document_to_client_queue(token, document_record_id)
    r.set(f"prev_split_page:{parent_document_id}", page)


# --------------------------------------------------------
# ENDPOINTS
# --------------------------------------------------------
@app.route("/split", methods=["POST"])
def split_endpoint():
    """
    Receives a POST request with "transaction_id", "document_id" and "signed_get_url" in body.

    token -> client auth token.
    transactino_id -> id of transaction associated with the document.
    document_id -> id of document record in database.
    signed_get_url -> aws s3 bucket signed url for downloading the file.

    Offloads the heavy lifting to a thread in ThreadPoolExecutor.
    Responds with 200 status code.
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
        split_request,
        token,
        int(transaction_id),
        int(document_id),
        signed_get_url,
    )

    return jsonify({"message": "Splitting started"}), 202


@app.route("/process", methods=["POST"])
def process_endpoint():
    """
    Receives a POST request with "transaction_id", "document_id" and "signed_get_url" in body.

    token -> client auth token.
    document_id -> id of document record in database.
    signed_get_url -> aws s3 bucket signed url for downloading the file.

    Offloads the heavy lifting to a thread in ThreadPoolExecutor.
    Responds with 200 status code.
    """

    print("received request to process document...")
    data = request.get_json() or {}

    if "token" not in data:
        return jsonify({"error": "Missing 'token' in request body"}), 400
    if "document_id" not in data:
        return jsonify({"error": "Missing 'document_id' in request body"}), 400
    if "signed_get_url" not in data:
        return jsonify({"error": "Missing 'signed_get_url' in request body"}), 400

    token = data.get("token")
    document_id = data.get("document_id")
    signed_get_url = data.get("signed_get_url")

    if not isinstance(token, str):
        return jsonify({"error": "'token' must be a string"}), 400
    if not isinstance(document_id, str) and not isinstance(document_id, int):
        return jsonify({"error": "'document_id' must be a number"}), 400
    if not isinstance(signed_get_url, str):
        return jsonify({"error": "'signed_get_url' must be a string"}), 400

    executor.submit(
        process_request,
        token,
        int(document_id),
        signed_get_url,
    )

    return jsonify({"message": "Processing started"}), 202


# --------------------------------------------------------
# ENDPOINT HANDLERS
# --------------------------------------------------------
def split_request(
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
    document_pages = len(merged_doc)
    merged_doc.close()

    r.set(f"prev_split_page:{parent_document_id}", -1)

    for i in range(0, document_pages, 1):
        try:
            # extract page contents if not on redis.
            page_content = r.get(f"page_content:{parent_document_id}:{i}")
            if page_content is None:
                page_image = convert_pdf_page_to_image(merged_file_name, i)

                if page_image is None:
                    continue

                page_content = get_image_contents(page_image)
                r.set(f"page_content:{parent_document_id}:{i}", page_content)

            # extract contents of next pages if not on redis.
            # todo: parallel.
            content = f"<curr_page>{page_content}</curr_page>"
            for j in range(1, min(pages_to_append, document_pages - i), 1):
                next_page_content = r.get(f"page_content:{parent_document_id}:{i + j}")

                if next_page_content is None:
                    next_page_image = convert_pdf_page_to_image(merged_file_name, i + j)

                    if next_page_image is None:
                        continue

                    next_page_content = get_image_contents(next_page_image)
                    r.set(
                        f"page_content:{parent_document_id}:{i + j}", next_page_content
                    )

                content += f"<next_page_{j}>{next_page_content}</next_page_{j}>"

            print(f"page {i + 1} content:", content[:100] + "...")

            if i == 0:
                continue

            if is_first_page(tokenizer, model, content) or i == document_pages - 1:
                print(f"found first page: {i + 1}")

                # bg task...
                try:
                    threading.Thread(
                        target=handle_first_page,
                        args=(
                            i,
                            int(
                                cast(
                                    str, r.get(f"prev_split_page:{parent_document_id}")
                                )
                            )
                            + 1,
                            token,
                            transaction_id,
                            parent_document_id,
                            merged_file_name,
                        ),
                    ).start()
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)

    print("notifying web server that splitting parent document has been finished...")
    notify_for_finished_splitting(token, parent_document_id)

    # clear keys from redis.
    # content keys are deleted here but after processing,
    # since they will be processed anyway and that needs the contents.
    splitting_keys = list(r.scan_iter("prev_split_page:*"))
    if splitting_keys:
        r.delete(*splitting_keys)


def process_request(
    token: str,
    document_id: int,
    signed_get_url: str,
):
    """
    The actual worker function that handles the request data.
    (Runs in a separate thread from the main Flask thread.)
    """

    file_name = f"{document_id}.pdf"

    contents_exist = True
    for i in range(0, pages_to_append):
        if r.get(f"page_content:{document_id}:{i}") is None:
            contents_exist = False
            break

    content = None
    if not contents_exist:
        print(f"downloading source file {file_name}...")
        file_path = download_s3_file(signed_get_url, file_name)

        doc = fitz.open(file_path)
        document_pages = len(doc)
        doc.close()

        # extract page contents if not on redis.
        page_image = convert_pdf_page_to_image(file_name, 0)

        page_content = ""
        if page_image is not None:
            page_content = get_image_contents(page_image)
            r.set(f"page_content:{document_id}:{0}", page_content)

        # extract contents of next pages if not on redis.
        # todo: parallel.
        content = f"<curr_page>{page_content}</curr_page>"
        for j in range(1, min(pages_to_append, document_pages), 1):
            next_page_image = convert_pdf_page_to_image(file_name, j)

            next_page_content = ""
            if next_page_image is not None:
                next_page_content = get_image_contents(next_page_image)
                r.set(f"page_content:{document_id}:{j}", next_page_content)

            content += f"<next_page_{j}>{next_page_content}</next_page_{j}>"
    else:
        page_content = str(r.get(f"page_content:{document_id}:{0}"))

        # extract contents of next pages if not on redis.
        # todo: parallel.
        content = f"<curr_page>{page_content}</curr_page>"
        for j in range(1, pages_to_append, 1):
            next_page_content = str(r.get(f"page_content:{document_id}:{j}"))
            content += f"<next_page_{j}>{next_page_content}</next_page_{j}>"

    print(f"content:", content[:100] + "...")

    print("notifying web server to add document to client queue...")
    data = {}  # prcessing...
    notify_for_finished_processing(token, document_id, data)

    # clear keys from redis.
    content_keys = list(r.scan_iter("page_content:*"))
    if content_keys:
        r.delete(*content_keys)


if __name__ == "__main__":
    print("Starting server...")
    app.run(host="0.0.0.0", port=8000, threaded=True)
