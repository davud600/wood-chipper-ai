from redis.client import Redis
from redis.lock import Lock

import redis as r

from config.settings import REDIS_HOST, REDIS_PORT

redis: Redis = r.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)


def get_lock(lock_name: str, timeout: int = 10, blocking_timeout: int = 10) -> Lock:
    return redis.lock(lock_name, timeout=timeout, blocking_timeout=blocking_timeout)
