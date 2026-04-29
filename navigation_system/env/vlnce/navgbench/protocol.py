"""Length-prefixed pickle protocol used by the NavGBench subprocess adapter."""

from __future__ import annotations

import pickle
import struct
from typing import Any, Dict


def read_exact(stream: Any, size: int, *, eof_message: str) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(eof_message)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(stream: Any, payload: Dict[str, Any]) -> None:
    data = pickle.dumps(payload, protocol=4)
    stream.write(struct.pack(">Q", len(data)))
    stream.write(data)
    stream.flush()


def receive_message(stream: Any, *, eof_message: str) -> Dict[str, Any]:
    header = read_exact(stream, 8, eof_message=eof_message)
    size = struct.unpack(">Q", header)[0]
    return pickle.loads(read_exact(stream, size, eof_message=eof_message))


__all__ = ["receive_message", "send_message"]
