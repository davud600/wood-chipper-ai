import requests
import json


def create_document_record(
    token: str, transaction_id: int, parent_document_id: int | None = None
) -> tuple[int, str]:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"http://localhost:3001/api/common-document?transactionId={transaction_id}&parentDocumentId={parent_document_id}",
        headers=headers,
    )

    if not response.ok:
        raise Exception("Failed to create document record.")

    json_data = json.loads(response.text)
    document_record_id = json_data.get("commonDocumentRecord").get("id")
    signed_put_url = json_data.get("signedPutUrl")

    return document_record_id, signed_put_url


def add_document_to_client_queue(token: str, document_id: int):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"http://localhost:3001/api/common-document-processor/finished_uploading_sub_document?transactionId={transaction_id}&documentId={document_id}",
        headers=headers,
    )

    if not response.ok:
        print(response.text)
        raise Exception("Failed to notify server.")


def notify_for_finished_splitting(token: str, parent_document_id: int):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"http://localhost:3001/api/common-document-processor/finished_splitting_document?transactionId={transaction_id}&parentDocumentId={document_id}",
        headers=headers,
    )

    if not response.ok:
        print(response.text)
        raise Exception("Failed to notify server.")
