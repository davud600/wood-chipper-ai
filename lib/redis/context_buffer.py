import json

from config.settings import prev_pages_to_append, pages_to_append
from .config import redis

BUFFER_KEY = lambda doc_id: f"context_buffer:{doc_id}"
PROCESSED_KEY = lambda doc_id: f"context_processed:{doc_id}"
LOCK_KEY = lambda doc_id: f"context_lock:{doc_id}"


def get_buffer(id: int) -> list[int]:
    # get context buffer from redis.
    data = redis.get(BUFFER_KEY(id))
    return json.loads(data) if data else []  # type: ignore


def get_processed(id: int) -> set:
    # get processed context buffer from redis.
    data = redis.smembers(PROCESSED_KEY(id))
    return {int(x) for x in data} if data else set()  # type: ignore


def push(id: int, item: int):
    with redis.lock(LOCK_KEY(id), timeout=3):
        buffer = get_buffer(id)

        if item not in buffer:
            buffer.append(item)
            buffer.sort()
            redis.set(BUFFER_KEY(id), json.dumps(buffer))


def get_ready_items(id: int) -> list[int]:
    with redis.lock(LOCK_KEY(id), timeout=3):
        buffer = get_buffer(id)
        processed = get_processed(id)

        ready = []
        for prime in buffer:
            if prime in processed:
                continue

            context_keys = [
                prime + i for i in range(-prev_pages_to_append, pages_to_append + 1)
            ]
            if all(k in buffer for k in context_keys):
                # context = [self.buffer[k] for k in context_keys]
                ready.append(prime)

        return ready


def mark_processed(id: int, item: int):
    with redis.lock(LOCK_KEY(id), timeout=3):
        redis.sadd(PROCESSED_KEY(id), item)
        cleanup(id)


def get_prev_items(id: int, item: int):
    buffer = get_buffer(id)

    if item not in buffer:
        return []

    idx = buffer.index(item)
    start = max(0, idx - prev_pages_to_append)

    return buffer[start:idx]


def get_next_items(id: int, item: int):
    buffer = get_buffer(id)

    if item not in buffer:
        return []

    idx = buffer.index(item)
    end = idx + 1 + pages_to_append

    return buffer[idx + 1 : end]


def cleanup(id: int):
    buffer = get_buffer(id)
    processed = get_processed(id)

    min_required = min(
        (p - prev_pages_to_append for p in buffer if p not in processed),
        default=0,
    )

    new_buffer = [p for p in buffer if p >= min_required]
    redis.set(BUFFER_KEY(id), json.dumps(new_buffer))
