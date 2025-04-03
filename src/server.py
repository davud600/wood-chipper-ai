from flask import Flask, request, jsonify
from transformers import AutoTokenizer

import multiprocessing
import threading
import torch

# import redis

from src.api import process_request, split_request
from src.utils import SPLITTER_MODEL_PATH
from src.model.model import SplitterModel


# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
MAX_WORKERS = 1

app = Flask(__name__)
manager = multiprocessing.Manager()
image_queue = manager.Queue()
content_queue = manager.Queue()
# r = redis.Redis(host="localhost", port=6379, decode_responses=True)


# --------------------------------------------------------
# LOAD MODEL & TOKENIZER
# --------------------------------------------------------
print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", device="cuda")

print("loading model...")
model = SplitterModel().to("cuda")
model.eval()
model.load_state_dict(
    torch.load(SPLITTER_MODEL_PATH, weights_only=False, map_location="cuda")
)


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

    document_context = {
        "token": token,
        "transaction_id": transaction_id,
        "document_id": document_id,
        "signed_get_url": signed_get_url,
    }

    threading.Thread(
        target=split_request,
        args=(tokenizer, model, image_queue, content_queue, document_context),
    ).start()

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

    document_context = {
        "token": token,
        "document_id": document_id,
        "signed_get_url": signed_get_url,
    }

    threading.Thread(
        target=process_request,
        args=(image_queue, content_queue, document_context),
    ).start()

    return jsonify({"message": "Processing started"}), 202


if __name__ == "__main__":
    print("Starting server...")
    app.run(host="0.0.0.0", port=8000, threaded=True, use_reloader=False)
