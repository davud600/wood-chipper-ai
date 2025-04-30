import numpy as np
import multiprocessing
import fitz

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitz import open as Document

from type_defs.shared import DocumentContext, SharedQueues

from lib.redis.queues import encode_image_queue_item, shared_queue_push
from lib.doc_tools import convert_pdf_page_to_image


def start_img_producers(
    pages: int,
    document_context: DocumentContext,
    workers: int,
) -> list[multiprocessing.Process]:
    """
    Starts image producer subprocesses for page-level image generation.

    Each process extracts a page image and pushes it to the image queue.

    Parameters
    ----------
    pages : int
        Total number of document pages.

    document_context : DocumentContext
        Metadata for the current document.

    workers : int
        Max number of concurrent image workers.

    Returns
    -------
    list of multiprocessing.Process
        List of spawned image producer processes.
    """

    image_processes: list[multiprocessing.Process] = []
    document = fitz.open(document_context["file_path"])

    for page in range(pages):
        process = multiprocessing.Process(
            target=image_producer,
            args=(document, page, document_context),
        )

        process.start()
        image_processes.append(process)

        if len(image_processes) >= workers:
            for proc in image_processes:
                proc.join()

            image_processes = []

    document.close()

    return image_processes


def image_producer(document: "Document", page: int, document_context: DocumentContext):
    """
    Extracts a page image from a PDF and pushes it to Redis queue.

    Parameters
    ----------
    document : fitz.Document
        Open PDF document.

    page : int
        Page number to process.

    document_context : DocumentContext
        Metadata for the document being processed.
    """

    try:
        img = convert_pdf_page_to_image(
            str(document_context["file_name"]), page, document
        )

        if img is None:
            return

        print(f"[img] {page}")
        encoded_img = encode_image_queue_item(page, img)
        shared_queue_push(
            document_context["document_id"], SharedQueues.Images, encoded_img
        )
        # print(f"[img] page {page} queued.")
    except Exception as e:
        encoded_img = encode_image_queue_item(page, np.array([]))
        shared_queue_push(
            document_context["document_id"], SharedQueues.Images, encoded_img
        )
        # print("error in image producer:", e)
