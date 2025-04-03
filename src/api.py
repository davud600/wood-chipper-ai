import multiprocessing
import threading
import numpy as np
import fitz

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from model.model import SplitterModel

from src.utils import pages_to_append, split_arr
from src.custom_types import ImageQueue, ContentQueue, DocumentContext
from src.lib.s3 import upload_file_to_s3

# from src.lib.openai import request_data_points
from src.lib.document_records import (
    add_document_to_client_queue,
    create_document_record,
    notify_for_finished_processing,
)
from src.document_processor.index import (
    convert_pdf_page_to_image,
    create_sub_document,
    get_image_contents,
    is_first_page,
)


# --------------------------------------------------------
# MAIN FUNCTIONS.
# --------------------------------------------------------


def process_pages_pipeline(
    image_queue: ImageQueue,
    content_queue: ContentQueue,
    page_batches: list[list[int]],
    doc: fitz.open,
    document_context: DocumentContext,
    max_img_workers: int = 4,
    max_ocr_workers: int = 2,
):
    start_ocr_workers(image_queue, content_queue, document_context, max_ocr_workers)

    # IMAGE PRODUCERS
    image_processes: list[multiprocessing.Process] = []
    for batch in page_batches:
        for page in batch:
            process = multiprocessing.Process(
                target=image_producer,
                args=(image_queue, page, doc, document_context),
            )

            process.start()
            image_processes.append(process)

            if len(image_processes) >= max_img_workers:
                for proc in image_processes:
                    proc.join()

                image_processes = []

    for process in image_processes:
        process.join()

    for _ in range(max_ocr_workers):
        image_queue.put(None)

    return


def inference_worker(
    tokenizer: "PreTrainedTokenizer",
    model: "SplitterModel",
    content_queue: ContentQueue,
    document_context: DocumentContext,
):
    batch_size = pages_to_append + 1
    batch = []

    while True:
        try:
            item = content_queue.get()
        except Exception as e:
            print("Error fetching from queue:", e)
            continue

        if item is None:
            print("exiting processed queue consumer thread...")
            break

        batch.append(item)

        try:
            # print(f"processed queue consumer: {page}")

            if len(batch) < batch_size:
                continue

            content_queue.task_done()
            content_batch = ""

            for i, (page, content) in enumerate(batch[:batch_size]):
                content_batch += (
                    f"<curr_content>{content}</curr_content>..."
                    if i == 0
                    else f"<next_page_{i}>{content}</next_page_{i}>..."
                )

            # inference...
            print(f"processed queue consumer found batch: {content_batch}")

            if is_first_page(tokenizer, model, content_batch):
                page, _ = batch[0]
                print(
                    f"batch starting at page {page} qualifies as first page; calling handle_first_page."
                )

                # handle_first_page(idx, prev_split_page, page, document_context)
        except Exception as e:
            print(e)


def start_ocr_workers(
    image_queue: ImageQueue,
    content_queue: ContentQueue,
    document_context: DocumentContext,
    workers: int = 2,
):
    ctx = multiprocessing.get_context("spawn")
    ocr_processes = []

    for _ in range(workers):
        process = ctx.Process(
            target=ocr_worker,
            args=(image_queue, content_queue, document_context),
        )

        process.start()
        ocr_processes.append(process)


def ocr_worker(
    image_queue: ImageQueue,
    content_queue: ContentQueue,
    document_context: DocumentContext,
):
    while True:
        item = image_queue.get()

        if item is None:  # sentinel value.
            print("exitting (ocr) image queue consumer thread...")
            break

        page, image = item

        try:
            content = get_image_contents(image) if len(image) > 0 else ""
            print(f"page {page} content: {content[:20]}...")

            image_queue.task_done()
            content_queue.put((page, content))
        except Exception as e:
            print(f"page: {page}: {e}")


def image_producer(
    queue: ImageQueue, page: int, doc: fitz.open, document_context: DocumentContext
):
    img = convert_pdf_page_to_image(str(document_context["file_name"]), page, doc)
    queue.put((page, img if img is not None else np.array([])))

    print(f"page {page} queued.")


def handle_first_page(
    idx: int, prev_split_page: int, page: int, document_context: DocumentContext
):
    """
    prev_split_page 0-based -> previous first page (don't include in sub-doc).
    page -> 0-based, model predicted this as start of new document (include in sub-doc).
    token -> client auth token.
    transaction_id -> id of transaction associated with parent document.
    parent_document_id -> merged document record id.
    merged_file_name -> file name of merged document.
    idx -> local sub doc iterator.
    """

    print(f"creating sub document {prev_split_page} - {page}...")
    document_id, signed_put_url = create_document_record(
        str(document_context["token"]),
        {
            "transactionId": int(document_context["transaction_id"]),
            "parentDocumentId": int(document_context["parent_document_id"]),
            "originalFileName": f"{str(document_context["file_name"]).replace('.pdf', '')}-{idx}.pdf",
        },
    )

    # update redis keys for doc contents to be used later on processing step.
    # for i in range(page - prev_split_page):
    #     pc = r.get(f"page_content:{parent_document_id}:{i + prev_split_page + 1}")
    #
    #     print(
    #         f"page_content:{document_id}:{i} => page_content:{parent_document_id}:{i + prev_split_page + 1} => {str(pc)[:30]}..."
    #     )
    #     # r.set(
    #     #     f"page_content:{document_id}:{i}",
    #     #     str(pc),
    #     # )

    sub_document_path = create_sub_document(
        str(document_context["file_name"]), prev_split_page, page, document_id
    )

    upload_file_to_s3(signed_put_url, sub_document_path)
    add_document_to_client_queue(str(document_context["token"]), document_id)


# --------------------------------------------------------
# ENDPOINT HANDLERS
# --------------------------------------------------------
def split_request(
    tokenizer: "PreTrainedTokenizer",
    model: "SplitterModel",
    image_queue: ImageQueue,
    content_queue: ContentQueue,
    document_context: DocumentContext,
):
    """
    The actual worker function that handles the request data.
    (Runs in a separate thread from the main Flask thread.)
    """

    # print(f"\n#{parent_document_id} downloading source file...")
    # document_context["file_name"] = f"{parent_document_id}.pdf"
    # document_context["file_path"] = download_s3_file(document_context["signed_get_url"], document_context["file_name"])
    document_context["file_name"] = "merged_removed.pdf"
    document_context["file_path"] = f"./{document_context["file_name"]}"

    merged_doc = fitz.open(document_context["file_path"])
    document_pages = len(merged_doc)
    print(f"{document_context["file_path"]} pages: {document_pages}")

    # r.set(f"prev_split_page:{parent_document_id}", -1)
    page_batches = split_arr(list(range(document_pages)), pages_to_append + 1)
    print("page_batches", page_batches)

    # process pages pipeline
    try:
        # start processed pages consumer.
        inference_worker_thread = threading.Thread(
            target=inference_worker,
            args=(tokenizer, model, content_queue, document_context),
        )
        inference_worker_thread.start()

        process_pages_pipeline(
            image_queue, content_queue, page_batches, merged_doc, document_context
        )

        inference_worker_thread.join()
    except Exception as e:
        print(e)

    merged_doc.close()
    # notify_for_finished_splitting(token, parent_document_id)


def process_request(
    image_queue: ImageQueue,
    ContentQueue: ContentQueue,
    document_context: DocumentContext,
):
    """
    The actual worker function that handles the request data.
    (Runs in a separate thread from the main Flask thread.)
    """

    return

    # file_name = f"{document_id}.pdf"
    # print("\n")
    #
    # contents_exist = True
    # for i in range(0, pages_to_append):
    #     paget_content = r.get(f"page_content:{document_id}:{i}")
    #
    #     print(
    #         f"#{document_id} page {i} content:",
    #         f"{str(paget_content)[:30]}...",
    #     )
    #
    #     if paget_content is None:
    #         contents_exist = False
    #         break
    #
    # print(f"{file_name} - {contents_exist}")
    #
    # content = None
    # if not contents_exist:
    #     file_path = download_s3_file(signed_get_url, file_name)
    #
    #     doc = fitz.open(file_path)
    #     document_pages = len(doc)
    #
    #     page_image = convert_pdf_page_to_image(file_name, 0, doc)
    #     page_content = ""
    #
    #     if page_image is not None:
    #         page_content = get_image_contents(page_image)
    #         r.set(f"page_content:{document_id}:{0}", page_content)
    #
    #     content = get_formatted_page_content_from_file_or_redis(
    #         doc=doc,
    #         document_id=document_id,
    #         file_name=file_name,
    #         page=0,
    #         page_content=page_content,
    #         pages_to_append=pages_to_append,
    #         document_pages=document_pages,
    #         r=r,
    #         check_redis=False,
    #     )
    #
    #     doc.close()
    # else:
    #     page_content = r.get(f"page_content:{document_id}:{0}")
    #
    #     content = get_formatted_page_content_from_file_or_redis(
    #         document_id=document_id,
    #         file_name=file_name,
    #         page=0,
    #         page_content=str(page_content),
    #         pages_to_append=pages_to_append,
    #         r=r,
    #         check_redis=True,
    #     )
    #
    # data = request_data_points(content)
    # print("\ndata:\n", data)
    #
    # notify_for_finished_processing(token, document_id, data)
