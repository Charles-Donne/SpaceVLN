"""Ordered runtime registry for classes detected during the current episode."""

from typing import Iterable, Iterator, List, Union


class DetectedClassRegistry:
    """Maintains insertion-ordered unique class names for runtime mapping state."""

    def __init__(self, values: Iterable[str] = ()) -> None:
        self._items = {}
        for value in values:
            self.add(value)

    def add(self, value: str) -> None:
        normalized = str(value or "").strip()
        if normalized:
            self._items[normalized] = None

    def discard(self, value: str) -> None:
        normalized = str(value or "").strip()
        if normalized:
            self._items.pop(normalized, None)

    def clear(self) -> None:
        self._items.clear()

    def copy(self) -> "DetectedClassRegistry":
        return DetectedClassRegistry(self._items.keys())

    def to_list(self) -> List[str]:
        return list(self._items.keys())

    def __contains__(self, value: object) -> bool:
        return value in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items.keys())

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: Union[int, slice]):
        values = self.to_list()
        return values[index]

    def __repr__(self) -> str:
        return f"DetectedClassRegistry({self.to_list()})"
