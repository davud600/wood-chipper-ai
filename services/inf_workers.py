import numpy as np
import multiprocessing
import torch
import os

from transformers import AutoTokenizer, PreTrainedTokenizer
from splitter.model import FusionModel
from PIL import Image


from lib.redis.config import get_lock
from config.settings import (
    SPLITTER_MODEL_PATH,
    pages_to_append,
    prev_pages_to_append,
    max_chars,
    image_output_size,
)
from type_defs.shared import DocumentContext, SharedQueues

from lib.redis import redis
from lib.redis.queues import (
    shared_queue_pop,
    decode_content_queue_item,
)
from lib.doc_tools import create_sub_document
from lib.document_records import create_document_record, add_document_to_client_queue
from lib.s3 import upload_file_to_s3
from splitter.inference import is_first_page

debug_dir = f"debug_batches"
os.makedirs(debug_dir, exist_ok=True)


def available_batch(q: list[int], c: int, t: int, a: int, b: int):
    try:
        idx = q.index(c)
        if idx + b >= len(q) and t not in q:
            return False
        if idx - a < 0 and 0 not in q:
            return False

        return True
    except Exception as e:
        print(e)
        print(q)
        print(c)
        print(t)
        return False


def consecutive(q: list[int]):
    if not q:
        return True
    prev = q[0]
    for i in q[1:]:
        if i != prev + 1:
            return False
        prev = i
    return True


def start_inf_workers(
    document_context: DocumentContext,
    pages: int,
    workers: int,
) -> list[multiprocessing.Process]:
    """
    Starts inference worker processes.

    Parameters
    ----------
    document_context : DocumentContext
        Metadata for the document being processed.

    pages : int
        Total number of pages in the document.

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
            args=(pages, document_context),
        )

        process.start()
        inf_processes.append(process)

    return inf_processes


def inference_worker(
    num_pages: int,
    document_context: DocumentContext,
):
    """
    Main loop for the inference worker.

    Consumes OCR results, builds context windows, performs model inference,
    and detects new document boundaries.

    Parameters
    ----------
    num_pages : int
        Total number of pages in the document.

    document_context : DocumentContext
        Metadata for the document being processed.
    """

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    print("loading model...")
    model = FusionModel(image_size=image_output_size).to("cuda")
    model.load_state_dict(
        torch.load(SPLITTER_MODEL_PATH, weights_only=False, map_location="cuda")
    )
    model.eval()

    queue: list[tuple[int, np.ndarray]] = []
    curr_page: int = 0

    while True:
        item = shared_queue_pop(document_context["document_id"], SharedQueues.Contents)

        if item is None:
            while queue and curr_page < num_pages:
                # print(
                #     f"[inf queue finishing...] [curr_page: {curr_page}] {[page for page, _ in queue]}"
                # )

                with get_lock("prev_split_page_lock"), get_lock("sub_doc_count_lock"):
                    process_batch(
                        tokenizer,
                        model,
                        queue,
                        curr_page,
                        num_pages,
                        document_context,
                    )

                    # get queue idx of page == curr page to get image.
                    curr_index = 0
                    for i, (page, _) in enumerate(queue):
                        if page == curr_page:
                            curr_index = i

                    if curr_index >= 2:
                        queue.pop(curr_index - 2)

                    curr_page += 1

            queue = []
            curr_page = 0
            print(f"exitting (inf) content queue consumer thread...")
            break

        page, image = decode_content_queue_item(item)

        if page not in queue:
            queue.append((page, image))
            queue.sort()

        if not consecutive([page for page, _ in queue]) or not available_batch(
            [page for page, _ in queue],
            curr_page,
            num_pages,
            prev_pages_to_append,
            pages_to_append,
        ):
            # print(queue)
            continue

        # print(f"[inf queue] [curr_page: {curr_page}] {queue}")

        with get_lock("prev_split_page_lock"), get_lock("sub_doc_count_lock"):
            process_batch(
                tokenizer, model, queue, curr_page, num_pages, document_context
            )

            # get queue idx of page == curr page to get image.
            curr_index = 0
            for i, (page, _) in enumerate(queue):
                if page == curr_page:
                    curr_index = i

            # print(
            #     f"[queue] [curr_page: {curr_page}] [curr_index: {curr_index}] {queue}"
            # )
            # process_batch(queue, curr_page, pages, document_context)

            if curr_index >= 2:
                queue.pop(curr_index - 2)

            curr_page += 1


def process_batch(
    tokenizer: PreTrainedTokenizer,
    model: FusionModel,
    queue: list[tuple[int, np.ndarray]],
    curr_page: int,
    num_pages: int,
    document_context: DocumentContext,
):
    """
    Performs inference on a contextual window of pages.

    Extracts image and text content around a target page,
    runs the prediction model, and updates Redis state if
    a new document is detected.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer used for text encoding.

    model : FusionModel
        Fusion model for inference.

    queue : list of tuple
        List of (page number, image array) from the shared queue.

    curr_page : int
        Page index being evaluated.

    num_pages : int
        Total number of pages in the document.

    document_context : DocumentContext
        Metadata for the document being processed.
    """

    # print(f"(inf document #{document_context["document_id"]}) consumer: {queue}")

    # get queue idx of page == curr page to get image.
    page_idx = 0
    for i, (page, _) in enumerate(queue):
        if page == curr_page:
            page_idx = i

    images: list[np.ndarray] = []
    content_batch = ""

    for offset in range(-prev_pages_to_append, pages_to_append + 1):
        image = np.zeros(image_output_size, dtype=np.uint8)
        content = ""
        idx = page_idx + offset

        if 0 <= idx < len(queue):
            page, image = queue[idx]

            # content
            raw = redis.get(f"page_content:{document_context["document_id"]}:{page}")
            content = raw.decode("utf-8") if raw else ""  # type: ignore

        # image
        images.append(image)

        if offset == 0:
            tag = "curr_page"
        elif offset < 0:
            tag = f"prev_page_{-offset}"
        else:
            tag = f"next_page_{offset}"

        if tag == "curr_page":
            char_limit = max_chars["curr_page"]
        elif "prev_page" in tag:
            char_limit = max_chars["prev_page"]
        elif "next_page" in tag:
            char_limit = max_chars["next_page"]
        else:
            char_limit = 512

        content_batch += f"<{tag}>{content[:char_limit]}</{tag}>"

    found_first_page = True
    offset = 0

    # print("content_batch", content_batch)
    print("\ncurr_page", curr_page)

    if curr_page < num_pages - 1:
        # debug: save images to disk.
        # for idx, img in enumerate(images):
        #     page_number = (
        #         queue[page_idx + idx - prev_pages_to_append][0]
        #         if 0 <= page_idx + idx - prev_pages_to_append < len(queue)
        #         else curr_page
        #     )
        #     debug_path = os.path.join(debug_dir, f"page_{page_number:03d}.png")
        #     Image.fromarray(img).save(debug_path)

        # inference...
        found_first_page, offset = is_first_page(
            tokenizer, model, content_batch, images
        )

    print("found_first_page", found_first_page)

    if found_first_page:
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
        print(f"detected new document: {prev_split_page} - {curr_page}")

        # update prev_split_page & sub_doc_count.
        redis.set(
            prev_split_key,
            (curr_page).to_bytes(4, byteorder="big", signed=True),
        )
        redis.set(
            sub_doc_key,
            (sub_doc_count + 1).to_bytes(4, byteorder="big", signed=False),
        )

        # handle_first_page(
        #     sub_doc_count, prev_split_page, curr_page, offset, document_context
        # )


def handle_first_page(
    sub_doc_count: int,
    prev_split_page: int,
    page: int,
    offset: int,
    document_context: DocumentContext,
):
    """
    Handles splitting of a detected new document.

    Updates Redis, creates a new document record, and uploads
    the split sub-document to S3.

    Parameters
    ----------
    sub_doc_count : int
        Number of sub-documents created so far.

    prev_split_page : int
        Start page of the last detected document segment.

    page : int
        Current page detected as a new document.

    offset : int
        Adjustment for current document segment length.

    document_context : DocumentContext
        Metadata for the parent document.
    """

    print(f"creating sub document {prev_split_page} - {page}...")
    document_id, signed_put_url = create_document_record(
        str(document_context["token"]),
        {
            "transactionId": int(document_context["transaction_id"]),
            "parentDocumentId": int(document_context["document_id"]),
            "originalFileName": f"{str(document_context["file_name"]).replace('.pdf', '')}-{sub_doc_count}.pdf",
        },
    )

    # update redis keys for doc contents to be used later on processing step.
    for i in range(page - prev_split_page - offset):
        raw = redis.get(
            f"page_content:{document_context["document_id"]}:{i + prev_split_page + offset}"
        )
        page_content: str = raw.decode("utf-8") if raw else ""  # type: ignore

        print(
            f"page_content:{document_context["document_id"]}:{i + prev_split_page + offset} => {str(page_content)[:10]} => page_content:{document_id}:{i}"
        )

        redis.set(
            f"page_content:{document_id}:{i}",
            page_content,
        )

    sub_document_path = create_sub_document(
        str(document_context["file_name"]), prev_split_page, page - offset, document_id
    )

    upload_file_to_s3(signed_put_url, sub_document_path)
    add_document_to_client_queue(str(document_context["token"]), document_id)
