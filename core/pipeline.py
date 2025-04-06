from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitz import open as Document

from config import img_workers, ocr_workers, inf_workers, ocr_batch_size
from type_defs import DocumentContext, SharedQueues

from lib.redis import redis
from lib.redis.queues import shared_queue_push
from services import start_img_producers, start_ocr_workers, start_inf_workers


def process_pages_pipeline(
    document: "Document",
    pages: int,
    document_context: DocumentContext,
    max_img_workers: int = img_workers,
    max_ocr_workers: int = ocr_workers,
    max_inf_workers: int = inf_workers,
    ocr_batch_size: int = ocr_batch_size,
):
    redis.set(
        f"prev_split_page:{document_context['document_id']}",
        (-1).to_bytes(4, byteorder="big", signed=True),
    )
    redis.set(
        f"sub_doc_count:{document_context['document_id']}",
        (0).to_bytes(4, byteorder="big", signed=False),
    )

    ocr_processes = start_ocr_workers(document_context, max_ocr_workers, ocr_batch_size)
    img_processes = start_img_producers(
        document, pages, document_context, max_img_workers
    )
    inf_processes = start_inf_workers(document_context, pages, max_inf_workers)

    for process in img_processes:
        process.join()

    for _ in ocr_processes:
        shared_queue_push(document_context["document_id"], SharedQueues.Images, None)

    for process in ocr_processes:
        process.join()

    for _ in inf_processes:
        shared_queue_push(document_context["document_id"], SharedQueues.Contents, None)

    for process in inf_processes:
        process.join()

    return
