import time

from type_defs.shared import SharedQueues

from .config import redis

max_image_queue_size = 32


def push(id: int | str, queue: SharedQueues, packed: bytes | None):
    if queue == SharedQueues.Images:
        while redis.llen(f"{queue.value}-{id}") >= max_image_queue_size:  # type: ignore
            time.sleep(0.001)

    if packed is None:
        redis.rpush(f"{queue.value}-{id}", b"__STOP__")
    else:
        redis.rpush(f"{queue.value}-{id}", packed)


def pop(id: int | str, queue: SharedQueues) -> bytes | None:
    raw = redis.blpop([f"{queue.value}-{id}"])

    if raw is None or raw[1] == b"__STOP__":  # type: ignore
        return None

    return raw[1]  # type: ignore
