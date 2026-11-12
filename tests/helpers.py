import hashlib
import math
import re

_WORD = re.compile(r"[a-z0-9]+")


def fake_embedding(text: str, dims: int = 64) -> list[float]:
    """A unit vector from hashed words. Shared words -> higher cosine."""
    vector = [0.0] * dims
    for word in _WORD.findall((text or "").lower()):
        if len(word) < 3:
            continue
        bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % dims
        vector[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]
