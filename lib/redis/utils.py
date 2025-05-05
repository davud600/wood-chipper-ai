import io
import numpy as np

from .config import redis


def get_page_content(id: int, page: int) -> str:
    raw = redis.get(f"page_content:{id}:{page}")
    return raw.decode("utf-8") if raw else ""  # type: ignore


def set_page_content(id: int, page: int, content: str):
    redis.set(f"page_content:{id}:{page}", content.encode("utf-8"))


def get_page_image(id: int, page: int) -> np.ndarray:
    raw = redis.get(f"page_image:{id}:{page}")
    if raw is None:
        raise ValueError(f"Missing image for doc {id}, page {page}")

    return np.load(io.BytesIO(raw), allow_pickle=False)  # type: ignore


def set_page_image(id: int, page: int, image: np.ndarray):
    buf = io.BytesIO()
    np.save(buf, image, allow_pickle=False)
    redis.set(f"page_image:{id}:{page}", buf.getvalue())


def encode_page_number(page: int) -> bytes:
    return page.to_bytes(4, byteorder="big", signed=True)


def decode_page_number(payload: bytes) -> int:
    return int.from_bytes(payload, byteorder="big", signed=True)
