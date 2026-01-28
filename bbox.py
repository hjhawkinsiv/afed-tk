from typing import Union


class BoundingBox:
    def __init__(self, left: int, top: int, right: int, bottom: int):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    @property
    def area(self) -> int:
        return (self.right - self.left) * (self.bottom - self.top)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    def center(self):
        return (self.right - self.left) // 2, (self.bottom - self.top) // 2

    def intersection(self, other) -> Union["BoundingBox", None]:
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        return BoundingBox(left, top, right, bottom) if left < right and top < bottom else None

    def scaled_down(self, scale: Union[int, float]) -> "BoundingBox":
        return BoundingBox(
            round(self.left / scale),
            round(self.top / scale),
            round(self.right / scale),
            round(self.bottom / scale)
        )

    def scaled_up(self, scale: Union[int, float]) -> "BoundingBox":
        return BoundingBox(
            round(self.left * scale),
            round(self.top * scale),
            round(self.right * scale),
            round(self.bottom * scale)
        )

    def translated(self, dx, dy) -> "BoundingBox":
        return BoundingBox(self.left + dx, self.top + dy, self.right + dx, self.bottom + dy)

    def __eq__(self, other: "BoundingBox") -> bool:
        return (self.left == other.left and
                self.top == other.top and
                self.right == other.right and
                self.bottom == other.bottom)

    def __ge__(self, other: "BoundingBox") -> bool:
        return (self.left <= other.left and
                self.top <= other.top and
                self.right >= other.right and
                self.bottom >= other.bottom)

    def __gt__(self, other: "BoundingBox") -> bool:
        return (self.left < other.left and
                self.top < other.top and
                self.right > other.right and
                self.bottom > other.bottom)

    def __le__(self, other: "BoundingBox") -> bool:
        return (self.left > other.left and
                self.top > other.top and
                self.right < other.right and
                self.bottom < other.bottom)

    def __lt__(self, other: "BoundingBox") -> bool:
        return (self.left >= other.left and
                self.top >= other.top and
                self.right <= other.right and
                self.bottom <= other.bottom)

    def __ne__(self, other: "BoundingBox") -> bool:
        return (self.left != other.left and
                self.top != other.top and
                self.right != other.right and
                self.bottom != other.bottom)

    def __repr__(self):
        return f"{BoundingBox.__name__}(left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom})"
