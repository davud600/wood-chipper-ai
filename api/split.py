import fitz

from config import pages_to_append
from type_defs import DocumentContext

from core import process_pages_pipeline
from lib.s3 import download_s3_file


def split_request(
    document_context: DocumentContext,
):
    """
    The actual worker function that handles the request data.
    (Runs in a separate thread from the main Flask thread.)
    """

    print(f"\n#{document_context["document_id"]} downloading source file...")
    document_context["file_name"] = f"{document_context["document_id"]}.pdf"
    document_context["file_path"] = download_s3_file(
        str(document_context["signed_get_url"]), document_context["file_name"]
    )

    merged_doc = fitz.open(document_context["file_path"])
    document_pages = len(merged_doc)
    print(f"{document_context["file_path"]} pages: {document_pages}")

    # r.set(f"prev_split_page:{parent_document_id}", -1)

    try:
        process_pages_pipeline(
            document=merged_doc,
            pages=min(pages_to_append + 1, document_pages),
            document_context=document_context,
        )
    except Exception as e:
        print(e)

    merged_doc.close()
    # notify_for_finished_splitting(token, parent_document_id)
