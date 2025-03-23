from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import torch

import csv  # temp: debugging.

from model import ClassifierModel
from utils import pages_to_append, max_length


# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
MAX_WORKERS = 2

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


# --------------------------------------------------------
# LOAD MODEL & TOKENIZER
# --------------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", device="cuda")

print("Loading model...")
model = ClassifierModel().to("cuda")
model.eval()
model.load_state_dict(
    torch.load("../model/model.pth", weights_only=False, map_location="cuda")
)


# --------------------------------------------------------
# ENDPOINT - SPLIT
# --------------------------------------------------------
@app.route("/split", methods=["POST"])
def generate_conversation_endpoint():
    """
    Receives a POST request with "documentId" and "signedGetUrl" in body.

    documentId -> id of document record in database.
    seignedGetUrl -> aws s3 bucket signed url for downloading the file.

    Offloads the heavy lifting to a thread in ThreadPoolExecutor.
    Responds with 200 status code, actual response is sent via web socket event.
    """

    # data = request.get_json() or {}
    #
    # if not data or "contents" not in data:
    #     return jsonify({"error": "Missing 'contents' in request body"}), 400
    #
    # contents = data.get("contents")
    #
    # if not isinstance(contents, list) or not all(
    #     isinstance(item, str) for item in contents
    # ):
    #     return jsonify({"error": "'contents' must be a list of strings"}), 400

    contents: list[str] = []
    with open("../merged.txt", mode="r", encoding="utf-8") as file:
        rows = csv.reader(file)
        for row in rows:
            contents += [row[0]]

    future = executor.submit(process_split_request, contents)
    return jsonify({"splitting_indices": future.result()})


def process_split_request(contents: list[str]) -> list[int]:
    """
    The actual worker function that handles the request data.
    (Runs in a separate thread from the main Flask thread.)
    """

    splitting_indices: list[int] = []
    for i, page_content in enumerate(contents):
        # contents batch.
        content = f"<curr_page>{page_content}</curr_page>"
        for j in range(1, min(pages_to_append, len(contents) - i), 1):
            content += f"<next_page_{j}>{contents[i + j]}</next_page_{j}>"

        print(f"page {i + 1} content:", content[:100] + "...")

        # tokenize batch.
        tokenized = tokenizer(
            [content],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        print(tokenized.input_ids)

        features = tokenized.input_ids.to("cuda")

        # inference.
        is_first_page = False
        with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=torch.float16):
            page_logits, _ = model(features)
            page_class = torch.argmax(page_logits, dim=1).item()

            # print("page_logits.shape:", page_logits.shape)

            # print(f"page {i + 1} class:", page_class)
            # print("page_logits:", page_logits)

            if torch.argmax(page_logits[0]):
                is_first_page = True

        # append to splitting_indices...
        if is_first_page:
            splitting_indices += [i]

    return splitting_indices


if __name__ == "__main__":
    print("Starting server...")
    app.run(host="0.0.0.0", port=8000, threaded=True)
