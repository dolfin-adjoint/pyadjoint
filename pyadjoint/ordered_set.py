from typing import Hashable, Iterable


class OrderedSet(set):
    def __init__(self, iterable: Iterable = ()):
        """
        An OrderedSet is a set that maintains the insertion order
        of its elements.

        Args:
          iterable: An iterable with which to initialise the set elements.
        """

        self._elements = []
        seen = set()

        for element in iterable:
            if element in seen:
                continue

            self._elements.append(element)
            seen.add(element)

        super().__init__(self._elements)

    def add(self, obj: Hashable):
        """Add obj to this set."""

        if obj in self:
            return

        super().add(obj)
        self._elements.append(obj)

    def remove(self, obj: Hashable):
        """Remove obj from this set, it must be a member."""

        if obj not in self:
            raise KeyError(obj)

        super().remove(obj)
        self._elements.remove(obj)

    def discard(self, obj: Hashable):
        """Remove obj from this set if it is present."""

        if obj not in self:
            return

        self.remove(obj)

    def __iter__(self):
        return iter(self._elements)

    def union(self, *others: Iterable):
        """Return a new set with elements from this set and others."""

        ret = OrderedSet(self)

        for other in others:
            for element in other:
                ret.add(element)

        return ret

    def __or__(self, other: set):
        if not isinstance(other, set):
            raise TypeError

        return self.union(other)

    def difference(self, *others: Iterable):
        """Return a new set with elements from this set
        that are not in others.
        """

        ret = OrderedSet(self)

        for other in others:
            for element in other:
                ret.discard(element)

        return ret

    def __sub__(self, other: set):
        if not isinstance(other, set):
            raise TypeError

        return self.difference(other)

    def intersection(self, *others: Iterable):
        """Return a new set with elements common to this set
        and all others.
        """

        ret = OrderedSet()

        for element in self:
            for other in others:
                if element not in other:
                    break
            else:
                ret.add(element)

        return ret

    def __and__(self, other: set):
        if not isinstance(other, set):
            raise TypeError

        return self.intersection(other)

    def symmetric_difference(self, other: Iterable):
        """Return a new set with elements either in this set or other,
        but not both.
        """

        ret = OrderedSet()

        for element in self:
            if element not in other:
                ret.add(element)

        for element in other:
            if element not in self:
                ret.add(element)

        return ret

    def __xor__(self, other: set):
        if not isinstance(other, set):
            raise TypeError

        return self.symmetric_difference(other)
