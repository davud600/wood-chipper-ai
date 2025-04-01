from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from typing import cast

import concurrent.futures
import threading
import redis
import torch
import fitz

# import csv  # temp: debugging.

from src.utils import SPLITTER_MODEL_PATH, DELETE_REDIS_KEYS_TIMEOUT, pages_to_append
from src.model.model import SplitterModel
from src.api.s3 import download_s3_file, upload_file_to_s3
from src.api.openai import request_data_points
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
    prev_split_page: int,
    page: int,
    offset: int,
    token: str,
    transaction_id: int,
    parent_document_id: int,
    merged_file_name: str,
    idx: int,
):
    """
    prev_split_page 0-based -> previous first page (don't include in sub-doc).
    page -> 0-based, model predicted this as start of new document (include in sub-doc).
    token -> client auth token.
    transaction_id -> id of transaction associated with parent document.
    parent_document_id -> merged document record id.
    merged_file_name -> file name of merged document.
    idx -> local sub doc iterator.
    """

    print(f"creating sub document {prev_split_page} - {page}...")
    document_id, signed_put_url = create_document_record(
        token,
        {
            "transactionId": transaction_id,
            "parentDocumentId": parent_document_id,
            "originalFileName": f"{merged_file_name.replace('.pdf', '')}-{idx}.pdf",
        },
    )

    # update redis keys for doc contents to be used later on processing step.
    for i in range(page - prev_split_page):
        pc = r.get(f"page_content:{parent_document_id}:{i + prev_split_page + 1}")

        print(
            f"page_content:{document_id}:{i} => page_content:{parent_document_id}:{i + prev_split_page + 1} => {str(pc)[:30]}..."
        )
        r.set(
            f"page_content:{document_id}:{i}",
            str(pc),
        )

    sub_document_path = create_sub_document(
        merged_file_name, prev_split_page, page, document_id
    )

    upload_file_to_s3(signed_put_url, sub_document_path)
    add_document_to_client_queue(token, document_id)
    r.set(f"prev_split_page:{parent_document_id}", page + offset)


def clear_keys_from_redis(document_id: int, split: bool = False):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_split_page = None

        if split:
            future_split_page = executor.submit(
                lambda: (lambda keys: r.delete(*keys) if keys else None)(
                    list(r.scan_iter(f"prev_split_page:{document_id}"))
                )
            )

        future_contents = executor.submit(
            lambda: (lambda keys: r.delete(*keys) if keys else None)(
                list(r.scan_iter(f"page_content:{document_id}:*"))
            )
        )

        try:
            if split and future_split_page:
                future_split_page.result(timeout=DELETE_REDIS_KEYS_TIMEOUT)

            future_contents.result(timeout=DELETE_REDIS_KEYS_TIMEOUT)
        except concurrent.futures.TimeoutError:
            print("Timeout: clearing keys took too long.")


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

    print(f"\n#{parent_document_id} downloading source file...")
    merged_file_name = f"{parent_document_id}.pdf"
    merged_file_path = download_s3_file(signed_get_url, merged_file_name)

    merged_doc = fitz.open(merged_file_path)
    document_pages = len(merged_doc)
    print("document_pages:", document_pages)
    merged_doc.close()

    r.set(f"prev_split_page:{parent_document_id}", -1)

    sub_docs = 0
    for i in range(0, document_pages, 1):
        try:
            page_content = r.get(f"page_content:{parent_document_id}:{i}")
            if page_content is None:
                page_image = convert_pdf_page_to_image(merged_file_name, i)

                if page_image is None:
                    continue

                page_content = get_image_contents(page_image)
                r.set(f"page_content:{parent_document_id}:{i}", page_content)

            content = f"<curr_page>{page_content}</curr_page>"
            for j in range(1, min(pages_to_append + 1, document_pages - i), 1):
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

            print(
                f"#{parent_document_id} page {i} content:",
                str(page_content)[:30] + "...",
            )

            if i == 0:
                continue

            first_page, offset = is_first_page(tokenizer, model, content)

            if first_page or i == document_pages - 1:
                sub_docs += 1
                print(f"\n#{parent_document_id} found first page: {i}")

                try:
                    threading.Thread(
                        target=handle_first_page,
                        args=(
                            int(
                                cast(
                                    str, r.get(f"prev_split_page:{parent_document_id}")
                                )
                            ),
                            i if i == document_pages - 1 else i - 1,
                            offset,
                            token,
                            transaction_id,
                            parent_document_id,
                            merged_file_name,
                            sub_docs,
                        ),
                    ).start()
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)

    notify_for_finished_splitting(token, parent_document_id)
    clear_keys_from_redis(parent_document_id, True)


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
    print("\n")

    contents_exist = True
    for i in range(0, pages_to_append):
        paget_content = r.get(f"page_content:{document_id}:{i}")

        print(
            f"#{document_id} page {i} content:",
            f"{str(paget_content)[:30]}...",
        )

        if paget_content is None:
            contents_exist = False
            break

    print(f"{file_name} - {contents_exist}")

    content = None
    if not contents_exist:
        file_path = download_s3_file(signed_get_url, file_name)

        doc = fitz.open(file_path)
        document_pages = len(doc)
        doc.close()

        page_image = convert_pdf_page_to_image(file_name, 0)

        page_content = ""
        if page_image is not None:
            page_content = get_image_contents(page_image)
            r.set(f"page_content:{document_id}:{0}", page_content)

        content = f"<curr_page>{page_content}</curr_page>"

        for j in range(1, min(pages_to_append + 1, document_pages), 1):
            next_page_image = convert_pdf_page_to_image(file_name, j)

            next_page_content = ""
            if next_page_image is not None:
                next_page_content = get_image_contents(next_page_image)
                r.set(f"page_content:{document_id}:{j}", next_page_content)

            content += f"<next_page_{j}>{next_page_content}</next_page_{j}>"
    else:
        page_content = str(r.get(f"page_content:{document_id}:{0}"))
        content = f"<curr_page>{page_content}</curr_page>"

        for j in range(1, pages_to_append + 1, 1):
            next_page_content = str(r.get(f"page_content:{document_id}:{j}"))
            content += f"<next_page_{j}>{next_page_content}</next_page_{j}>"

    data = request_data_points(content)
    print("data:", data)

    notify_for_finished_processing(token, document_id, data)
    clear_keys_from_redis(document_id)


if __name__ == "__main__":
    print("Starting server...")
    app.run(host="0.0.0.0", port=8000, threaded=True)
