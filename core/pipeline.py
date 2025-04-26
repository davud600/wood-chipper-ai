from config.settings import img_workers, ocr_workers, inf_workers, ocr_batch_size
from type_defs.shared import DocumentContext, SharedQueues

from lib.redis import redis
from lib.redis.queues import shared_queue_push
from services import start_img_producers, start_ocr_workers, start_inf_workers


def process_pages_pipeline(
    pages: int,
    document_context: DocumentContext,
    max_img_workers: int = img_workers,
    max_ocr_workers: int = ocr_workers,
    max_inf_workers: int = inf_workers,
    ocr_batch_size: int = ocr_batch_size,
):
    """
    Orchestrates the document processing pipeline for a given number of pages.

    Initializes Redis tracking keys and starts separate multiprocessing workers
    for image generation, OCR, and inference tasks. Ensures that the workers are
    joined in order and that signals are sent via shared queues to indicate
    the end of work for downstream stages.

    Parameters
    ----------
    pages : int
        The total number of pages in the document.

    document_context : DocumentContext
        A dictionary of contextual information for the document being processed,
        including token, document ID, and paths.

    max_img_workers : int, optional
        Maximum number of concurrent image producer workers to launch.

    max_ocr_workers : int, optional
        Maximum number of concurrent OCR workers to launch.

    max_inf_workers : int, optional
        Maximum number of concurrent inference workers to launch.

    ocr_batch_size : int, optional
        Number of images processed per batch during OCR.

    Returns
    -------
    None
    """

    redis.set(
        f"prev_split_page:{document_context['document_id']}",
        (0).to_bytes(4, byteorder="big", signed=True),
    )
    redis.set(
        f"sub_doc_count:{document_context['document_id']}",
        (0).to_bytes(4, byteorder="big", signed=False),
    )

    inf_processes = start_inf_workers(document_context, max_inf_workers, pages)
    ocr_processes = start_ocr_workers(document_context, max_ocr_workers, ocr_batch_size)
    img_processes = start_img_producers(pages, document_context, max_img_workers)

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
