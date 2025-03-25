from collections.abc import Hashable, Iterable, Iterator, MutableSet, Set


class OrderedSet(MutableSet):
    def __init__(self, iterable: Iterable[Hashable] = ()):
        """
        An OrderedSet is a set that maintains the insertion order
        of its elements.

        Args:
          iterable: An iterable with which to initialise the set elements.
        """

        self._elements: dict[Hashable, None] = {}

        for element in iterable:
            self._elements[element] = None

    def __contains__(self, obj: Hashable) -> bool:
        return obj in self._elements

    def __len__(self) -> int:
        return len(self._elements)

    def copy(self) -> "OrderedSet":
        """Return a shallow copy of this set"""

        ret = type(self)()
        ret._elements = self._elements.copy()

        return ret

    def add(self, obj: Hashable) -> None:
        """Add obj to this set."""

        self._elements[obj] = None

    def remove(self, obj: Hashable) -> None:
        """Remove obj from this set, it must be a member."""

        if obj not in self:
            raise KeyError(obj)

        del self._elements[obj]

    def discard(self, obj: Hashable) -> None:
        """Remove obj from this set if it is present."""

        if obj not in self:
            return

        self.remove(obj)

    def __iter__(self) -> Iterator:
        return iter(self._elements.keys())

    def union(self, *others: Iterable[Hashable]) -> "OrderedSet":
        """Return a new set with elements from this set and others."""

        ret = self.copy()

        for other in others:
            for element in other:
                ret.add(element)

        return ret

    def __or__(self, other: Set) -> "OrderedSet":
        if not isinstance(other, Set):
            raise TypeError

        return self.union(other)

    def difference(self, *others: Iterable[Hashable]) -> "OrderedSet":
        """Return a new set with elements from this set
        that are not in others.
        """

        ret = self.copy()

        for other in others:
            for element in other:
                ret.discard(element)

        return ret

    def __sub__(self, other: Set) -> "OrderedSet":
        if not isinstance(other, Set):
            raise TypeError

        return self.difference(other)

    def intersection(self, *others: Iterable[Hashable]) -> "OrderedSet":
        """Return a new set with elements common to this set
        and all others.
        """

        ret = type(self)()

        for element in self:
            for other in others:
                if element not in other:
                    break
            else:
                ret.add(element)

        return ret

    def __and__(self, other: Set) -> "OrderedSet":
        if not isinstance(other, Set):
            raise TypeError

        return self.intersection(other)

    def symmetric_difference(self, other: Iterable[Hashable]) -> "OrderedSet":
        """Return a new set with elements either in this set or other,
        but not both.
        """

        ret = type(self)()

        for element in self:
            if element not in other:
                ret.add(element)

        for element in other:
            if element not in self:
                ret.add(element)

        return ret

    def __xor__(self, other: Set) -> "OrderedSet":
        if not isinstance(other, Set):
            raise TypeError

        return self.symmetric_difference(other)
