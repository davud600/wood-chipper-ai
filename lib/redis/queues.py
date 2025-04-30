import numpy as np
import msgpack
import time

from type_defs.shared import SharedQueues

from .config import redis

max_image_queue_size = 32


def encode_content_queue_item(page: int, image: np.ndarray) -> bytes:
    return msgpack.packb(  # type: ignore
        {
            "page": page,
            "shape": image.shape,
            "dtype": str(image.dtype),
            "data": image.tobytes(),
        },
        use_bin_type=True,
    )


def decode_content_queue_item(payload: bytes) -> tuple[int, np.ndarray]:
    unpacked = msgpack.unpackb(payload, raw=False)

    shape = tuple(unpacked["shape"])
    dtype = np.dtype(unpacked["dtype"])
    image = np.frombuffer(unpacked["data"], dtype=dtype).reshape(shape)

    return unpacked["page"], image


def encode_image_queue_item(page: int, image: np.ndarray) -> bytes:
    return msgpack.packb(  # type: ignore
        {
            "page": page,
            "shape": image.shape,
            "dtype": str(image.dtype),
            "data": image.tobytes(),
        },
        use_bin_type=True,
    )


def decode_image_queue_item(payload: bytes) -> tuple[int, np.ndarray]:
    unpacked = msgpack.unpackb(payload, raw=False)

    shape = tuple(unpacked["shape"])
    dtype = np.dtype(unpacked["dtype"])
    image = np.frombuffer(unpacked["data"], dtype=dtype).reshape(shape)

    return unpacked["page"], image


def shared_queue_push(id: int | str, queue: SharedQueues, packed: bytes | None):
    if queue == SharedQueues.Images:
        while redis.llen(f"{queue.value}-{id}") >= max_image_queue_size:  # type: ignore
            time.sleep(0.001)

    if packed is None:
        redis.rpush(f"{queue.value}-{id}", b"__STOP__")
    else:
        redis.rpush(f"{queue.value}-{id}", packed)


def shared_queue_pop(id: int | str, queue: SharedQueues) -> bytes | None:
    raw = redis.blpop([f"{queue.value}-{id}"])

    if raw is None or raw[1] == b"__STOP__":  # type: ignore
        return None

    return raw[1]  # type: ignore


def peek_queue_batch(id: int | str, queue: SharedQueues, n: int) -> list[bytes]:
    return redis.lrange(f"{queue.value}-{id}", 0, n - 1)  # type: ignore
