import fitz

from config.settings import pages_to_append
from type_defs.shared import DocumentContext

from core import process_pages_pipeline
from lib.redis import redis
from lib.s3 import download_s3_file
from lib.openai import request_data_points
from lib.document_records import notify_for_finished_processing


def process_request(
    document_context: DocumentContext,
):
    """
    Processes the document content and sends data points to downstream services.

    If Redis has cached page content, this is aggregated and sent to an
    OpenAI data point extraction pipeline. If not, the file is downloaded,
    and processing is done via `process_pages_pipeline`.

    Parameters
    ----------
    document_context : DocumentContext
        Dictionary containing metadata about the document, including
        token, document ID, and signed S3 URL.

    Returns
    -------
    None
    """

    contents: list[str] = []
    for i in range(0, pages_to_append):
        raw = redis.get(f"page_content:{document_context['document_id']}:{i}")
        page_content: str = raw.decode("utf-8") if raw else ""  # type: ignore

        # print(
        #     f"#{document_context['document_id']} page {i} content:",
        #     f"{page_content[:25]}...",
        # )
        contents += [page_content]

    document_context["file_name"] = f"{document_context['document_id']}.pdf"

    content_batch = ""
    if len(contents[0]) > 0:
        for i, content in enumerate(contents):
            content_batch += (
                f"<curr_content>{content}</curr_content>..."
                if i == 0
                else f"<next_page_{i}>{content}</next_page_{i}>..."
            )
    else:
        document_context["file_path"] = download_s3_file(
            str(document_context["signed_get_url"]), document_context["file_name"]
        )

        doc = fitz.open(document_context["file_path"])
        document_pages = len(doc)
        doc.close()

        try:
            process_pages_pipeline(
                pages=min(pages_to_append + 1, document_pages),
                document_context=document_context,
                max_inf_workers=0,
            )

        except Exception as e:
            print(e)

    data = request_data_points(content_batch)
    notify_for_finished_processing(
        str(document_context["token"]),
        int(document_context["document_id"]),
        data,
    )

    # delete redis keys.
    keys_to_delete = redis.keys(f"*:{document_context['document_id']}*")
    if keys_to_delete:
        redis.delete(*keys_to_delete)  # type: ignore
