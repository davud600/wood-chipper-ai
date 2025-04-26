import multiprocessing
import numpy as np
import os


from PIL import Image
from typing import Dict
from transformers import AutoTokenizer, PreTrainedTokenizer

from .context_buffer import ContextBuffer
from config.settings import (
    SPLITTER_MODEL_DIR,
    max_chars,
    image_output_size,
)
from type_defs.shared import DocumentContext, SharedQueues, InferWorkerState
from lib.redis import redis
from lib.redis.queues import (
    shared_queue_pop,
    decode_content_queue_item,
)
from lib.doc_tools import create_sub_document
from lib.document_records import create_document_record, add_document_to_client_queue
from lib.s3 import upload_file_to_s3
from splitter.inference import is_first_page
from splitter.model import FusionModel

debug_dir = f"debug_batches"
os.makedirs(debug_dir, exist_ok=True)

session_dirs = [
    int(d)
    for d in os.listdir(SPLITTER_MODEL_DIR)
    if os.path.isdir(os.path.join(SPLITTER_MODEL_DIR, d)) and d.isdigit()
]
session = max(session_dirs, default=0)

last_cut_was_separator = False


def start_inf_workers(
    document_context: DocumentContext,
    workers: int,
    pages: int,
) -> list[multiprocessing.Process]:
    """
    Starts inference worker processes.

    Parameters
    ----------
    document_context : DocumentContext
        Metadata for the document being processed.

    workers : int
        Number of inference workers to spawn.

    Returns
    -------
    list of multiprocessing.Process
        List of inference worker processes.
    """

    ctx = multiprocessing.get_context("spawn")
    inf_processes = []

    for _ in range(workers):
        process = ctx.Process(
            target=inference_worker,
            args=(document_context, pages),
        )

        process.start()
        inf_processes.append(process)

    return inf_processes


def inference_worker(document_context: DocumentContext, pages: int):
    """
    Main loop for the inference worker.

    Consumes OCR results, builds this pretty weird local context window,
    performs model inference, and detects new document boundaries.

    Parameters
    ----------
    document_context : DocumentContext
        Metadata for the document being processed.
    """

    # print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # print("loading model...")
    model = FusionModel(image_size=image_output_size).to("cuda")
    # load_best_weights(model, session, True)
    # model.load_state_dict(
    #     torch.load(SPLITTER_MODEL_PATH, weights_only=False, map_location="cuda")
    # )
    model.eval()

    ctx_buff = ContextBuffer()
    images: Dict[int, np.ndarray] = {}

    while True:
        item = shared_queue_pop(document_context["document_id"], SharedQueues.Contents)

        if item is None:
            for prime_page in ctx_buff.buffer:
                process_page(
                    document_context,
                    tokenizer,
                    model,
                    prime_page,
                    ctx_buff.get_prev_items(prime_page),
                    ctx_buff.get_next_items(prime_page),
                    images,
                    pages,
                )
                ctx_buff.mark_processed(prime_page)
            break

        page, image = decode_content_queue_item(item)
        ctx_buff.push(page)
        images[page] = image

        for prime_page in ctx_buff.get_ready_items():
            # print(f"processing page {page}")
            process_page(
                document_context,
                tokenizer,
                model,
                prime_page,
                ctx_buff.get_prev_items(prime_page),
                ctx_buff.get_next_items(prime_page),
                images,
                pages,
            )
            ctx_buff.mark_processed(prime_page)


def process_page(
    document_context: "DocumentContext",
    tokenizer: PreTrainedTokenizer,
    model: FusionModel,
    page: int,
    prev_pages,
    next_pages,
    images: Dict[int, np.ndarray],
    pages: int,
):
    """
    Run inference on a single page using surrounding context.

    This function builds the multimodal input (text and images) for a given page,
    calls the model to classify whether it's the first page of a new document,
    and updates Redis-tracked state if a new document boundary is detected.

    Parameters
    ----------
    document_context : DocumentContext
        Metadata and Redis keys for the current document stream.

    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer used for preparing text input for the LLM.

    model : FusionModel
        The fusion model that combines LLM and CNN outputs to classify first pages.

    page : int
        Current page number being evaluated (0-based).

    prev_pages : List[int]
        Page numbers used as backward context.

    next_pages : List[int]
        Page numbers used as forward context.

    images : Dict[int, np.ndarray]
        Dictionary mapping page numbers to grayscale image arrays.
    """

    global last_cut_was_separator

    # get images & contents of all pages and format them.
    content_batch = ""

    for i, prev_page in enumerate(prev_pages):
        prev_content = get_page_content(document_context, prev_page)
        content_batch += f"<prev_page_{len(prev_pages) - i}>{prev_content[:max_chars["prev_page"]]}</prev_page_{len(prev_pages) - i}>"

    image = images.get(page)
    content = get_page_content(document_context, page)
    content_batch += f"<curr_page>{content[:max_chars["curr_page"]]}</curr_page>"

    for i, next_page in enumerate(next_pages):
        next_content = get_page_content(document_context, next_page)
        content_batch += f"<next_page_{i + 1}>{next_content[:max_chars["next_page"]]}</next_page_{i + 1}>"

    # inference if not last or first page of doc.
    state = get_state(document_context)

    if page == pages - 1:
        handle_first_page(
            state["sub_doc_count"],
            state["prev_split_page"],
            page,
            0 if not last_cut_was_separator else 1,
            document_context,
        )
        return
    if page == 0:
        return

    # # debug: save images to disk.
    # debug_path = os.path.join(debug_dir, f"page_{page:03d}.png")
    # Image.fromarray(image).save(debug_path)  # type: ignore

    # (mostly) lease renewals edge case:
    # check content similarity between prime page and prev split page.
    # if too similar (almost identical) then skip.
    # prev_split_page_content = get_page_content(
    #     document_context, prev_split_page + 1
    # )
    # similarity = Levenshtein.ratio(prev_split_page_content, content)
    # if similarity > 0.985:
    #     return

    # inference...
    # print(f"[inference] page {page} - prev page {state["prev_split_page"]}")
    distance = (page - 1) - state["prev_split_page"]
    found_first_page, offset = is_first_page(
        tokenizer, model, content_batch, distance, image
    )

    if offset > 0:
        last_cut_was_separator = True

    # if first page call func.
    if found_first_page:
        handle_first_page(
            state["sub_doc_count"],
            state["prev_split_page"],
            page - 1,
            offset,
            document_context,
        )
        update_state(document_context, page + offset, state["sub_doc_count"] + 1)


def handle_first_page(
    count: int,
    start_page: int,
    end_page: int,
    offset: int,
    document_context: DocumentContext,
):
    """
    Handles splitting of a detected new document.

    Updates Redis, creates a new document record, and uploads
    the split sub-document to S3.

    Parameters
    ----------
    count : int
        Number of sub-documents created so far.

    start_page : int
        Start page of the last detected document segment.

    end_page : int
        Current page detected as a new document.

    offset : int
        Adjustment for current document segment length.

    document_context : DocumentContext
        Metadata for the parent document.
    """

    print(f"\ncreating sub document {start_page} - {end_page}...")
    document_id, signed_put_url = create_document_record(
        str(document_context["token"]),
        {
            "transactionId": int(document_context["transaction_id"]),
            "parentDocumentId": int(document_context["document_id"]),
            "originalFileName": f"{str(document_context["file_name"]).replace('.pdf', '')}-{count}.pdf",
        },
    )

    # update redis keys for doc contents to be used later on processing step.
    for i in range((end_page - start_page) + 1):
        raw = redis.get(
            f"page_content:{document_context["document_id"]}:{i + start_page}"
        )
        page_content: str = raw.decode("utf-8") if raw else ""  # type: ignore

        print(
            f"page_content:{document_context["document_id"]}:{i + start_page} ({str(page_content)[:10]}) => page_content:{document_id}:{i}"
        )

        redis.set(
            f"page_content:{document_id}:{i}",
            page_content,
        )

    sub_document_path = create_sub_document(
        str(document_context["file_name"]), start_page, end_page, document_id
    )

    upload_file_to_s3(signed_put_url, sub_document_path)
    add_document_to_client_queue(str(document_context["token"]), document_id)


def get_page_content(document_context: DocumentContext, page: int) -> str:
    raw = redis.get(f"page_content:{document_context["document_id"]}:{page}")
    return raw.decode("utf-8") if raw else ""  # type: ignore


def get_state(document_context: DocumentContext) -> InferWorkerState:
    prev_split_key = f"prev_split_page:{document_context['document_id']}"
    sub_doc_key = f"sub_doc_count:{document_context['document_id']}"

    prev_split_bytes = redis.get(prev_split_key)
    sub_doc_bytes = redis.get(sub_doc_key)

    if prev_split_bytes is None:
        raise ValueError(f"Missing Redis value for key: {prev_split_key}")
    if sub_doc_bytes is None:
        raise ValueError(f"Missing Redis value for key: {sub_doc_key}")

    prev_split_page = int.from_bytes(prev_split_bytes, byteorder="big", signed=True)  # type: ignore
    sub_doc_count = int.from_bytes(sub_doc_bytes, byteorder="big")  # type: ignore

    return {"prev_split_page": prev_split_page, "sub_doc_count": sub_doc_count}


def update_state(
    document_context: DocumentContext, prev_split_page: int, sub_doc_count: int
):
    prev_split_key = f"prev_split_page:{document_context['document_id']}"
    sub_doc_key = f"sub_doc_count:{document_context['document_id']}"

    redis.set(
        prev_split_key,
        prev_split_page.to_bytes(4, byteorder="big", signed=True),
    )
    redis.set(
        sub_doc_key,
        sub_doc_count.to_bytes(4, byteorder="big", signed=False),
    )
