import fitz

from config import pages_to_append
from type_defs import DocumentContext

from core import process_pages_pipeline
from lib.redis import redis
from lib.s3 import download_s3_file
from lib.openai import request_data_points
from lib.document_records import notify_for_finished_processing


def process_request(
    document_context: DocumentContext,
):
    """
    The actual worker function that handles the request data.
    (Runs in a separate thread from the main Flask thread.)
    """

    contents: list[str] = []
    for i in range(0, pages_to_append):
        raw = redis.get(f"page_content:{document_context["document_id"]}:{i}")
        page_content: str = raw.decode("utf-8") if raw else ""  # type: ignore

        print(
            f"#{document_context["document_id"]} page {i} content:",
            f"{page_content[:30]}...",
        )
        contents += [page_content]

    print(f"{document_context["file_name"]} - {len(contents) > 0}")

    content_batch = ""
    if len(contents) > 0:
        for i in range(len(contents)):
            content = redis.get(f"page_content:{document_context["document_id"]}:{i}")
            content_batch += (
                f"<curr_content>{content}</curr_content>..."
                if i == 0
                else f"<next_page_{i}>{content}</next_page_{i}>..."
            )
    else:
        file_path = download_s3_file(
            str(document_context["signed_get_url"]), str(document_context["file_name"])
        )

        doc = fitz.open(file_path)
        document_pages = len(doc)

        try:
            process_pages_pipeline(
                document=doc,
                pages=min(pages_to_append + 1, document_pages),
                document_context=document_context,
                max_inf_workers=0,
            )

            data = request_data_points(content_batch)
            print("\ndata:\n", data)

            notify_for_finished_processing(
                str(document_context["token"]),
                int(document_context["document_id"]),
                data,
            )
        except Exception as e:
            print(e)
