import multiprocessing

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitz import open as Document

from type_defs import DocumentContext, SharedQueues

from lib.redis.queues import encode_image_queue_item, shared_queue_push
from lib.doc_tools import convert_pdf_page_to_image


def start_img_producers(
    document: "Document",
    pages: int,
    document_context: DocumentContext,
    workers: int,
) -> list[multiprocessing.Process]:
    image_processes: list[multiprocessing.Process] = []

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

    return image_processes


def image_producer(document: "Document", page: int, document_context: DocumentContext):
    try:
        img = convert_pdf_page_to_image(
            str(document_context["file_name"]), page, document
        )

        if img is None:
            return

        encoded_img = encode_image_queue_item(img, page)
        shared_queue_push(
            document_context["document_id"], SharedQueues.Images, encoded_img
        )
        # print(f"page {page} queued.")
    except Exception as e:
        print("error in image producer:", e)
