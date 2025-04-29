import numpy as np

from typing import Dict, Tuple, Union

from enum import Enum

DocumentContext = Dict[str, Union[int, str]]

InferWorkerState = Dict[str, int]

DatasetMiniBatch = Dict[str, Union[list[str], list[int]]]
Dataset = list[DatasetMiniBatch]

FileContents = Dict[int, str]


class SharedQueues(Enum):
    Images = "image_queue"
    Contents = "content_queue"


ContentQueueItem = int | None


ImageQueueItem = Tuple[int, np.ndarray] | None


DatasetRow = tuple[str, int, int, str]
