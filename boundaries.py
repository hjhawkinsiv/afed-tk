import numpy as np

from math import ceil, floor


class BoundingCircle:
    def __init__(self, x: int, y: int, radius: int):
        self.x, self.y, self.radius = x, y, radius

    @property
    def center(self) -> tuple[int, int]:
        return self.x, self.y

    def contract_by(self, scale: int | float):
        self.radius = floor(self.radius / scale)
        
    def contracted_by(self, scale: int | float) -> "BoundingCircle":
        return BoundingCircle(self.x, self.y, floor(self.radius / scale))

    def dilate_by(self, scale: int | float):
        self.radius = ceil(self.radius * scale)

    def dilated_by(self, scale: int | float) -> "BoundingCircle":
        return BoundingCircle(self.x, self.y, ceil(self.radius * scale))

    def scale(self, distance: int):
        self.radius = self.radius + distance
    
    def scaled(self, distance: int) -> "BoundingCircle":
        return BoundingCircle(self.x, self.y, self.radius + distance)

    def __add__(self, distance: int) -> "BoundingCircle":
        return BoundingCircle(self.x, self.y, self.radius + distance)

    def __div__(self, scale: int | float) -> "BoundingCircle":
        return BoundingCircle(self.x, self.y, floor(self.radius / scale))

    def __mul__(self, scale: int | float) -> "BoundingCircle":
        return BoundingCircle(
            ceil(self.x * scale),
            ceil(self.y * scale),
            ceil(self.radius * scale),
            ceil(self.bottom * scale)
        )

    def __sub__(self, distance: int) -> "BoundingCircle":
        return BoundingCircle(self.x, self.y, self.radius - distance)

    def __eq__(self, other: "BoundingCircle") -> bool:
        return self.x == other.x and self.y == other.y and self.radius == other.radius

    def __ne__(self, other: "BoundingCircle") -> bool:
        return self.x != other.x or self.y != other.y or self.radius != other.radius

    def __repr__(self):
        return f"{BoundingCircle.__name__}[center=({self.x}, {self.y}), radius={self.radius}]"


class BoundingRect:
    def __init__(self, left: int, top: int, right: int, bottom: int):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom

    @staticmethod
    def minimum_containing(points: np.ndarray | list[tuple[int, int]]) -> BoundingRect:

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    @property
    def height(self):
        return self.bottom - self.top

    @property
    def center(self):
        return floor(self.bottom - self.top / 2), floor((self.right - self.left) / 2)

    @property
    def width(self):
        return self.right - self.left

    def contract_by(self, scale: int | float):
        self.left = floor(self.left / scale)
        self.top = floor(self.top / scale)
        self.right = floor(self.right / scale)
        self.bottom = floor(self.bottom / scale)
        
    def contracted_by(self, scale: int | float) -> "BoundingRect":
        return BoundingRect(
            floor(self.left / scale), 
            floor(self.top / scale),
            floor(self.right / scale), 
            floor(self.bottom / scale)
        )

    def dilate_by(self, scale: int | float):
        self.left = ceil(self.left * scale)
        self.top = ceil(self.top * scale)
        self.right = ceil(self.right * scale)
        self.bottom = ceil(self.bottom * scale)

    def dilated_by(self, scale: int | float) -> "BoundingRect":
        return BoundingRect(
            ceil(self.left * scale),
            ceil(self.top * scale),
            ceil(self.right * scale),
            ceil(self.bottom * scale)
        )

    def intersection(self, other) -> "BoundingRect | None":
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        return BoundingRect(left, top, right, bottom) if left < right and top < bottom else None

    def scale(self, dx: int | tuple[int, int], dy: int | tuple[int, int]):
        if isinstance(dx, tuple):
            self.left -= dx[0]
            self.right += dx[1]
        else:
            self.left -= dx
            self.right += dx

        if isinstance(dy, tuple):
            self.top -= dy[0]
            self.bottom += dy[1]
        else:
            self.top -= dy
            self.bottom += dy
    
    def scaled(self, dx: int | tuple[int, int], dy: int | tuple[int, int]) -> "BoundingRect":
        if isinstance(dx, tuple):
            l, r = self.left - dx[0], self.right + dx[1]
        else:
            l, r = self.left - dx, self.right + dx

        if isinstance(dy, tuple):
            t, b = self.top - dy[0], self.bottom + dy[1]
        else:
            t, b = self.top - dy, self.bottom + dy

        return BoundingRect(l, t, r, b)
    
    def translate(self, dx: int | tuple[int, int], dy: int | tuple[int, int]) -> "BoundingRect":
        if isinstance(dx, tuple):
            self.left += dx[0]
            self.right += dx[1]
        else:
            self.left += dx
            self.right += dx

        if isinstance(dy, tuple):
            self.top += dy[0]
            self.bottom += dy[1]
        else:
            self.top += dy
            self.bottom += dy

    def translated(self, dx: int | tuple[int, int], dy: int | tuple[int, int]) -> "BoundingRect":
        if isinstance(dx, tuple):
            l, r = self.left + dx[0], self.right + dx[1]
        else:
            l, r = self.left + dx, self.right + dx

        if isinstance(dy, tuple):
            t, b = self.top + dy[0], self.bottom + dy[1]
        else:
            t, b = self.top + dy, self.bottom + dy

        return BoundingRect(l, t, r, b)

    def __add__(self, scale: int | tuple[int, int] | tuple[int, int, int, int]) -> "BoundingRect":
        if isinstance(scale, tuple):
            if len(scale) == 2:
                l, r = self.left + scale[0], self.right + scale[0]
                t, b = self.top + scale[1], self.bottom + scale[1]
            else:
                l, r = self.left + scale[0], self.right + scale[1]
                t, b = self.top + scale[2], self.bottom + scale[3]
        else:
            l, r = self.left + scale, self.right + scale
            t, b = self.top + scale, self.bottom + scale

        return BoundingRect(l, t, r, b)

    def __div__(self, scale: int | float) -> "BoundingRect":
        return BoundingRect(
            floor(self.left / scale),
            floor(self.top / scale),
            floor(self.right / scale),
            floor(self.bottom / scale)
        )

    def __mul__(self, scale: int | float) -> "BoundingRect":
        return BoundingRect(
            ceil(self.left * scale),
            ceil(self.top * scale),
            ceil(self.right * scale),
            ceil(self.bottom * scale)
        )

    def __sub__(self, scale: int | tuple[int, int] | tuple[int, int, int, int]) -> "BoundingRect":
        if isinstance(scale, tuple):
            if len(scale) == 2:
                l, r = self.left - scale[0], self.right - scale[0]
                t, b = self.top - scale[1], self.bottom - scale[1]
            else:
                l, r = self.left - scale[0], self.right - scale[1]
                t, b = self.top - scale[2], self.bottom - scale[3]
        else:
            l, r = self.left - scale, self.right - scale
            t, b = self.top - scale, self.bottom - scale

        return BoundingRect(l, t, r, b)

    def __and__(self, other) -> "BoundingRect | None":
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        return BoundingRect(left, top, right, bottom) if left < right and top < bottom else None

    def __or__(self, other) -> "BoundingRect | None":
        left = min(self.left, other.left)
        top = min(self.top, other.top)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)

        return BoundingRect(left, top, right, bottom) if left < right and top < bottom else None

    def __eq__(self, other: "BoundingRect") -> bool:
        return (self.left == other.left and
                self.top == other.top and
                self.right == other.right and
                self.bottom == other.bottom)

    def __ge__(self, other: "BoundingRect") -> bool:
        return (self.left <= other.left and
                self.top <= other.top and
                self.right >= other.right and
                self.bottom >= other.bottom)

    def __gt__(self, other: "BoundingRect") -> bool:
        return (self.left < other.left and
                self.top < other.top and
                self.right > other.right and
                self.bottom > other.bottom)

    def __le__(self, other: "BoundingRect") -> bool:
        return (self.left > other.left and
                self.top > other.top and
                self.right < other.right and
                self.bottom < other.bottom)

    def __lt__(self, other: "BoundingRect") -> bool:
        return (self.left >= other.left and
                self.top >= other.top and
                self.right <= other.right and
                self.bottom <= other.bottom)

    def __ne__(self, other: "BoundingRect") -> bool:
        return (self.left != other.left or
                self.top != other.top or
                self.right != other.right or
                self.bottom != other.bottom)

    def __repr__(self):
        return f"{BoundingRect.__name__}[left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom}]"


