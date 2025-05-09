from flask import Flask, request, jsonify
from datetime import datetime

import logging
import multiprocessing
import threading

from api import process_request, split_request

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
MAX_WORKERS = 1

app = Flask(__name__)
manager = multiprocessing.Manager()

logging.basicConfig(
    filename="gunicorn_output.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# --------------------------------------------------------
# ENDPOINTS
# --------------------------------------------------------
@app.route("/healthcheck", methods=["GET"])
def healthcheck_endpoint():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"[{timestamp}] healthcheck ok")
    return jsonify(), 200


@app.route("/split", methods=["POST"])
def split_endpoint():
    """
    Handles document splitting requests.

    This endpoint receives a JSON payload containing a client token,
    transaction ID, document ID, and a signed AWS S3 URL. The request
    is validated and then processed in a separate thread.

    Parameters
    ----------
    None

    Returns
    -------
    flask.Response
        A JSON response with status code 202 if the request is accepted,
        or 400 if any required fields are missing or invalid.
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

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"[{timestamp}] split request")
    threading.Thread(
        target=split_request,
        args=(document_context,),
    ).start()

    return jsonify({"message": "Splitting started"}), 202


@app.route("/process", methods=["POST"])
def process_endpoint():
    """
    Handles document processing requests.

    This endpoint receives a JSON payload containing a client token,
    document ID, and a signed AWS S3 URL. The request is validated
    and then processed in a separate thread.

    Parameters
    ----------
    None

    Returns
    -------
    flask.Response
        A JSON response with status code 202 if the request is accepted,
        or 400 if any required fields are missing or invalid.
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

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"[{timestamp}] process request")
    threading.Thread(
        target=process_request,
        args=(document_context,),
    ).start()

    return jsonify({"message": "Processing started"}), 202


if __name__ == "__main__":
    logging.info("Starting server...")
    app.run(host="0.0.0.0", port=8001, threaded=True, use_reloader=False)
