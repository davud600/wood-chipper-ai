from config.settings import prev_pages_to_append, pages_to_append


class ContextBuffer:
    """
    Keep track of processed items.

    Weird ctx thing to "process" items in order but only when the "next" and
    "prev" pages are also present. And an item from the local ctx
    is not removed if it is needed as the "prev" page of the next one/s.
    """

    def __init__(
        self, prev_n: int = prev_pages_to_append, next_n: int = pages_to_append
    ):
        self.prev_n: int = prev_n
        self.next_n: int = next_n
        self.buffer: list[int] = []
        self.processed = set()

    def push(self, item: int):
        self.buffer.append(item)
        self.buffer.sort()

    def get_ready_items(self) -> list[int]:
        ready = []
        for prime in self.buffer:
            if prime in self.processed:
                continue

            context_keys = [prime + i for i in range(-self.prev_n, self.next_n + 1)]
            if all(k in self.buffer for k in context_keys):
                context = [self.buffer[k] for k in context_keys]
                ready.append(prime)

        return ready

    def mark_processed(self, item: int):
        self.processed.add(item)
        self.cleanup()

    def get_prev_items(self, item: int):
        if item not in self.buffer:
            return []
        idx = self.buffer.index(item)
        start = max(0, idx - self.prev_n)
        return self.buffer[start:idx]

    def get_next_items(self, item: int):
        if item not in self.buffer:
            return []
        idx = self.buffer.index(item)
        end = idx + 1 + self.next_n
        return self.buffer[idx + 1 : end]

    def cleanup(self):
        min_required = min(
            (p - self.prev_n for p in self.buffer if p not in self.processed),
            default=0,
        )
        self.buffer = [p for p in self.buffer if p >= min_required]
