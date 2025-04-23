import fitz

from type_defs.shared import DocumentContext

from core import process_pages_pipeline
from lib.redis import redis
from lib.s3 import download_s3_file
from lib.document_records import notify_for_finished_splitting


def split_request(
    document_context: DocumentContext,
):
    """
    Splits the document into individual pages and initiates processing.

    Downloads the source PDF file from an S3 URL, counts the number of pages,
    and invokes the `process_pages_pipeline`. Upon completion, it notifies
    the document record service and clears relevant Redis cache entries.

    Parameters
    ----------
    document_context : DocumentContext
        Dictionary containing metadata about the document, including
        token, transaction ID, document ID, and signed S3 URL.

    Returns
    -------
    None
    """

    print(f"\n#{document_context['document_id']} downloading source file...")
    document_context["file_name"] = f"{document_context['document_id']}.pdf"
    document_context["file_path"] = download_s3_file(
        str(document_context["signed_get_url"]), document_context["file_name"]
    )

    merged_doc = fitz.open(document_context["file_path"])
    document_pages = len(merged_doc)
    merged_doc.close()
    print(f"{document_context['file_path']} pages: {document_pages}")

    try:
        process_pages_pipeline(
            pages=document_pages,
            document_context=document_context,
        )
    except Exception as e:
        print(e)

    notify_for_finished_splitting(
        str(document_context["token"]), int(document_context["document_id"])
    )

    # delete redis keys.
    keys_to_delete = redis.keys(f"*:{document_context['document_id']}*")
    if keys_to_delete:
        redis.delete(*keys_to_delete)  # type: ignore
