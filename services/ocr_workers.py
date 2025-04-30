import multiprocessing
import numpy as np

from type_defs.shared import DocumentContext, SharedQueues
from lib.redis import redis
from lib.redis.queues import (
    encode_content_queue_item,
    decode_image_queue_item,
    shared_queue_pop,
    shared_queue_push,
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

    images_batch: list[tuple[int, np.ndarray]] = []

    while True:
        item = shared_queue_pop(document_context["document_id"], SharedQueues.Images)

        if item is None:  # sentinel value.
            if len(images_batch) > 0:
                process_batch(document_context, images_batch)
                images_batch = []

            break

        page, image = decode_image_queue_item(item)
        images_batch += [(page, image)]
        print(f"[ocr] {page}")

        if len(images_batch) >= batch_size:
            process_batch(document_context, images_batch)
            images_batch = []


def process_batch(
    document_context: DocumentContext, batch: list[tuple[int, np.ndarray]]
):
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

    try:
        contents = get_image_batch_contents([img for _, img in batch])

        for i, (page_number, image) in enumerate(batch):
            redis.set(
                f"page_content:{document_context['document_id']}:{page_number}",
                contents[i],
            )

            encoded = encode_content_queue_item(page_number, image)
            shared_queue_push(
                document_context["document_id"], SharedQueues.Contents, encoded
            )
    except Exception as e:
        print(f"Batch processing error (pages {[p for p, _ in batch]}): {e}")
