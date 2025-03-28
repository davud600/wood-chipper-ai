from typing import Dict, TypedDict
from enum import Enum

type DatasetMiniBatch = Dict[str, list[str] | list[int]]
type Dataset = list[DatasetMiniBatch]

type FileContents = Dict[int, str]


type DatasetRow = tuple[str, int, int, str]


class EdgeCaseFile(TypedDict):
    case: str


type EdgeCaseFiles = Dict[str, EdgeCaseFile]


class EdgeCases(Enum):
    START = r"start\((\d+)\)"
    ALIAS = r"alias\((\d+)\)"
    DELETE = "delete"
    AGREEMENT = "agreement"
    SUBLEASE = "sublease"
