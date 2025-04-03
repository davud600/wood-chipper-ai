import numpy as np

from typing import Dict, TypedDict, Tuple
from queue import Queue
from enum import Enum

type DocumentContext = Dict[str, int | str]

type DatasetMiniBatch = Dict[str, list[str] | list[int]]
type Dataset = list[DatasetMiniBatch]

type FileContents = Dict[int, str]


ContentQueueItem = Tuple[int, str] | None


class ContentQueue(Queue[ContentQueueItem]):
    pass


ImageQueueItem = Tuple[int, np.ndarray] | None


class ImageQueue(Queue[ImageQueueItem]):
    pass


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
