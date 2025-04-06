import multiprocessing
import numpy as np

from type_defs import DocumentContext, SharedQueues

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
    images_batch: list[tuple[int, np.ndarray]] = []

    while True:
        item = shared_queue_pop(document_context["document_id"], SharedQueues.Images)

        if item is None:  # sentinel value.
            if len(images_batch) > 0:
                process_batch(document_context, images_batch)
                images_batch = []

            print("exitting (ocr) image queue consumer thread...")
            break

        page, image = decode_image_queue_item(item)
        images_batch += [(page, image)]
        # print(
        #     f"(ocr) #{document_context["document_id"]} {[page for page, _ in images_batch]}"
        # )

        if len(images_batch) >= batch_size:
            process_batch(document_context, images_batch)
            images_batch = []


def process_batch(
    document_context: DocumentContext, batch: list[tuple[int, np.ndarray]]
):
    if not batch:
        return

    try:
        contents = get_image_batch_contents([img for _, img in batch])

        for i, (page_number, _) in enumerate(batch):
            redis.set(
                f"page_content:{document_context['document_id']}:{page_number}",
                contents[i],
            )

            encoded = encode_content_queue_item(page_number)
            shared_queue_push(
                document_context["document_id"], SharedQueues.Contents, encoded
            )
    except Exception as e:
        print(f"Batch processing error (pages {[p for p, _ in batch]}): {e}")
