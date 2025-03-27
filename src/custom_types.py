from typing import Dict
from enum import Enum

type DatasetMiniBatch = Dict[str, list[str] | list[int]]
type Dataset = list[DatasetMiniBatch]

type FileContents = Dict[int, str]


class EdgeCases(Enum):
    START = r"start\((\d+)\)"
    ALIAS = r"alias\((\d+)\)"
    DELETE = "delete"
    AGREEMENT = "agreement"
    SUBLEASE = "sublease"


class DocumentType(Enum):
    UNKNOWN = 0
    ORIGINAL_LEASE = 1
    LEASE_RENEWAL = 2
    CLOSING_DOCUMENT = 3
    SUBLEASE = 4
    RENOVATION_ALTERATION_DOCUMENT = 5
    PROPRIETARY_LEASE = 7
    PURCHASE_APPLICATION = 8
    REFINANCE_DOCUMENT = 9
    TENANT_CORRESPONDENCE = 10
    TRANSFER_DOCUMENT = 11
    SUBLEASE_RENEWAL = 11
    TRANSFER_OF_TITLE = 11
