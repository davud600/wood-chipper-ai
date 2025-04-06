import requests
import json

from typing import Dict

from config import API_URL


def create_document_record(token: str, data: Dict[str, str | int]) -> tuple[int, str]:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{API_URL}/api/common-document", headers=headers, json=data
    )

    if not response.ok:
        raise Exception("Failed to create document record.")

    json_data = json.loads(response.text)
    document_record_id = json_data.get("commonDocument").get("id")
    signed_put_url = json_data.get("signedPutUrl")

    return document_record_id, signed_put_url


def update_document_record(
    token: str, id: int, data: Dict[str, str | int]
) -> tuple[int, str]:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.put(
        f"{API_URL}/api/common-document/{id}", headers=headers, json=data
    )

    if not response.ok:
        raise Exception("Failed to update document record.")

    json_data = json.loads(response.text)
    document_record_id = json_data.get("commonDocument").get("id")
    signed_put_url = json_data.get("signedPutUrl")

    return document_record_id, signed_put_url


def add_document_to_client_queue(token: str, document_id: int):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{API_URL}/api/common-document-processor/finished_uploading_sub_document?documentId={document_id}",
        headers=headers,
    )

    if not response.ok:
        print(response.text)
        raise Exception("Failed to notify server.")


def notify_for_finished_splitting(token: str, parent_document_id: int):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{API_URL}/api/common-document-processor/finished_splitting_document?parentDocumentId={parent_document_id}",
        headers=headers,
    )

    if not response.ok:
        print(response.text)
        raise Exception("Failed to notify server.")


def notify_for_finished_processing(token: str, document_id: int, data: Dict[str, str]):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{API_URL}/api/common-document-processor/finished_processing_document?documentId={document_id}",
        headers=headers,
        json=data,
    )

    if not response.ok:
        print(response.text)
        raise Exception("Failed to notify server.")
