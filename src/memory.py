from collections import deque
from time import time

class Timeline:
    def __init__(self, max_items=200):
        self.buf = deque(maxlen=max_items)

    def add(self, items):
        ts = time()
        for it in items:
            self.buf.append({"ts": ts, **it})

    def get_corpus(self):
        return [e["text"] for e in self.buf if e.get("conf", 0) >= 0.6]