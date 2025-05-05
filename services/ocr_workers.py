import multiprocessing
import lib.redis.context_buffer as ctx_buff
import lib.redis.queue as redis_queue

from type_defs.shared import DocumentContext, SharedQueues
from lib.redis.utils import (
    get_page_image,
    set_page_content,
    decode_page_number,
    encode_page_number,
)
from lib.doc_tools import get_image_batch_contents


def start_ocr_workers(
    document_context: DocumentContext, workers: int, batch_size: int
) -> list[multiprocessing.Process]:
    """
    Starts OCR worker processes using multiprocessing.

    Parameters
    ----------
    document_context : DocumentContext
        Metadata for the document being processed.

    workers : int
        Number of OCR workers to spawn.

    batch_size : int
        Number of images to process in each batch.

    Returns
    -------
    list of multiprocessing.Process
        List of running OCR worker processes.
    """

    ctx = multiprocessing.get_context("spawn")
    ocr_processes = []

    for _ in range(workers):
        process = ctx.Process(
            target=ocr_worker,
            args=(document_context, batch_size),
        )

        process.start()
        ocr_processes.append(process)

    return ocr_processes


def ocr_worker(document_context: DocumentContext, batch_size: int):
    """
    OCR processing loop.

    Continuously consumes image batches from Redis queue, performs OCR,
    and pushes the results into the content queue.

    Parameters
    ----------
    document_context : DocumentContext
        Metadata for the current document.

    batch_size : int
        Maximum number of images to process in one batch.
    """

    batch: list[int] = []

    while True:
        item = redis_queue.pop(document_context["document_id"], SharedQueues.Images)

        if item is None:  # sentinel value.
            if len(batch) > 0:
                process_batch(document_context, batch)
                batch = []

            break

        page = decode_page_number(item)
        batch += [page]
        print(f"[ocr] {page}")

        if len(batch) >= batch_size:
            process_batch(document_context, batch)
            batch = []


def process_batch(document_context: DocumentContext, batch: list[int]):
    """
    Processes a batch of page images through OCR.

    Extracts content, stores it in Redis, and pushes to content queue.

    Parameters
    ----------
    document_context : DocumentContext
        Metadata for the current document.

    batch : list of tuple
        Each tuple contains (page number, image array).
    """

    if not batch:
        return

    document_id = int(document_context["document_id"])

    try:
        contents = get_image_batch_contents(
            [get_page_image(document_id, page_number) for page_number in batch]
        )

        # debugging
        print(f"[ocr] batch: {[pn for pn in batch]}")

        for i, page_number in enumerate(batch):
            set_page_content(document_id, page_number, contents[i])
            ctx_buff.push(document_id, page_number)

        for page_number in ctx_buff.get_ready_items(document_id):
            encoded = encode_page_number(page_number)
            redis_queue.push(document_id, SharedQueues.Contents, encoded)

    except Exception as e:
        print(f"[ocr] batch processing error (pages {[p for p in batch]}): {e}")
