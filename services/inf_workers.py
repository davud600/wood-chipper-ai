import multiprocessing
import torch

from transformers import AutoTokenizer, PreTrainedTokenizer
from training.model.model import SplitterModel

from lib.redis.config import get_lock
from config import pages_to_append, SPLITTER_MODEL_PATH
from type_defs import DocumentContext, SharedQueues

from lib.redis import redis
from lib.redis.queues import (
    shared_queue_pop,
    decode_content_queue_item,
)
from lib.doc_tools import create_sub_document
from lib.document_records import create_document_record, add_document_to_client_queue
from lib.s3 import upload_file_to_s3
from inference import is_first_page


def get_consecutive_window(pages: list[int], size: int) -> list[int] | None:
    pages = sorted(set(pages))  # safe to sort every time unless perf is critical

    for i in range(len(pages) - size + 1):
        window = pages[i : i + size]
        if all(b - a == 1 for a, b in zip(window, window[1:])):
            return window
    return None


def start_inf_workers(
    document_context: DocumentContext,
    pages: int,
    workers: int,
) -> list[multiprocessing.Process]:
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
    pages: int,
    document_context: DocumentContext,
):
    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/longformer-base-4096", device="cuda"
    )

    print("loading model...")
    model = SplitterModel().to("cuda")
    model.eval()
    model.load_state_dict(
        torch.load(SPLITTER_MODEL_PATH, weights_only=False, map_location="cuda")
    )

    batch_size = pages_to_append + 1
    batch = []

    while True:
        item = shared_queue_pop(document_context["document_id"], SharedQueues.Contents)

        if item is None:
            while len(batch) > 0:
                # print("clearing rest of batch in inf worker...")
                with get_lock("prev_split_page_lock"), get_lock("sub_doc_count_lock"):
                    process_batch(tokenizer, model, batch, pages, document_context)
                    batch.pop(0)

            print(f"exitting (inf) content queue consumer thread...")
            break

        page = decode_content_queue_item(item)

        if page not in batch:
            batch.append(page)
            batch.sort()
            # print(
            #     f"[batch update] doc {document_context["document_id"]} batch: {sorted(batch)}"
            # )

        window = get_consecutive_window(batch, batch_size)

        if window:
            # print(
            #     f"(inf) content queue (document #{document_context["document_id"]}) consumer: {window}"
            # )
            with get_lock("prev_split_page_lock"), get_lock("sub_doc_count_lock"):
                process_batch(tokenizer, model, batch, pages, document_context)

                batch.pop(0)
                window = get_consecutive_window(batch, batch_size)


def process_batch(
    tokenizer: PreTrainedTokenizer,
    model: SplitterModel,
    batch: list[int],
    pages: int,
    document_context: DocumentContext,
):
    print(f"(inf document #{document_context["document_id"]}) consumer: {batch}")

    page = batch[0]
    content_batch = ""

    for i in range(len(batch)):
        raw = redis.get(f"page_content:{document_context["document_id"]}:{page + i}")
        content = raw.decode("utf-8") if raw else ""  # type: ignore

        content_batch += (
            f"<curr_page>{content}</curr_page>..."
            if i == 0
            else f"<next_page_{i}>{content}</next_page_{i}>..."
        )

    # if detected new doc or last page.
    found_first_page = False
    if page < pages - 1:
        # inference...
        found_first_page, offset = is_first_page(tokenizer, model, content_batch)
    else:
        found_first_page, offset = True, 0

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
        print(f"detected new document: {prev_split_page} - {page + offset}")

        # update prev_split_page & sub_doc_count.
        redis.set(
            prev_split_key,
            (page + offset).to_bytes(4, byteorder="big", signed=True),
        )
        redis.set(
            sub_doc_key,
            (sub_doc_count + 1).to_bytes(4, byteorder="big", signed=False),
        )

        # handle_first_page(
        #     sub_doc_count, prev_split_page, page, offset, document_context
        # )


def handle_first_page(
    sub_doc_count: int,
    prev_split_page: int,
    page: int,
    offset: int,
    document_context: DocumentContext,
):
    """
    sub_doc_count -> counter for the number of sub docs created until now from the parent doc.
    prev_split_page 0-based -> previous first page (don't include in sub-doc).
    page -> 0-based, model predicted this as start of new document (include in sub-doc).
    document_context -> parent document metadata.
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
    for i in range(page - prev_split_page):
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
        str(document_context["file_name"]), prev_split_page, page, document_id
    )

    upload_file_to_s3(signed_put_url, sub_document_path)
    add_document_to_client_queue(str(document_context["token"]), document_id)
