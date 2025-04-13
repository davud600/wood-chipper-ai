import numpy as np

from typing import Dict, Tuple

from enum import Enum

type DocumentContext = Dict[str, int | str]

type DatasetMiniBatch = Dict[str, list[str] | list[int]]
type Dataset = list[DatasetMiniBatch]

type FileContents = Dict[int, str]


class SharedQueues(Enum):
    Images = "image_queue"
    Contents = "content_queue"


ContentQueueItem = int | None


ImageQueueItem = Tuple[int, np.ndarray] | None


type DatasetRow = tuple[str, int, int, str]
