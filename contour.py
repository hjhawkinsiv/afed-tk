import datetime
import numpy as np
import os

from pathlib import Path
from typing import Union

from bbox import BoundingBox
from utilities import export_to_mp4

FramePadding = Union[int, tuple[int, int]]
PixelPadding = Union[int, tuple[int, int], tuple[int, int, int, int]]


class TemporalContour:
    def __init__(self, points: np.ndarray):
        """Creates a new contour from a list of points

        ## Parameters

        * `points` - a collection of (frame, row, column) triples
        """
        self.points = points
        self.frames = np.sort(np.unique(points[:, 0]))
        self.number_of_frames = len(self.frames)
        self.number_of_points = len(points)
        self._region = None

    @staticmethod
    def load_points(contour_points_file):
        """Creates a `TemporanContour` instance defined by points saved using `np.save` with `allow_pickle` set to `False`

        ## Parameters

        * `contour_points_file:` - source of the points

        """
        return TemporalContour(np.load(contour_points_file, allow_pickle=False))

    @property
    def bounding_box(self) -> BoundingBox:
        """The `BoundingBox` containing **all** points in the contour"""
        if self._region is None:
            max_y, max_x, min_y, min_x = 0, 0, 2147483647, 2147483647

            for frame in self.frames:
                mask_candidates = self.points[:, 0] == frame
                points = self.points[mask_candidates, 1:]
                y_values, x_values = points[:, 0], points[:, 1]

                max_y = max(max_y, np.max(y_values))
                max_x = max(max_x, np.max(x_values))
                min_y = min(min_y, np.min(y_values))
                min_x = min(min_x, np.min(x_values))

            self._region = BoundingBox(min_x, min_y, max_x, max_y)

        return self._region

    @property
    def corners(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """The top left and bottom right corners of the contour"""
        return (
            (self.bounding_box.top, self.bounding_box.left),
            (self.bounding_box.bottom, self.bounding_box.right)
        )

    @property
    def height(self) -> int:
        """The height of the bounding box containing **all** points in the contour."""
        return self.bounding_box.height

    @property
    def width(self) -> int:
        """The height of the bounding box containing **all** points in the contour."""
        return self.bounding_box.width

    def bounding_boxes_by_frame(self) -> list[BoundingBox]:
        """A dictionary of bounding boxes keyed by frame"""
        regions = []

        for frame in self.frames:
            points = self.points[self.points[:, 0] == frame, 1:]
            y_values, x_values = points[:, 0], points[:, 1]
            max_y = np.max(y_values)
            max_x = np.max(x_values)
            min_y = np.min(y_values)
            min_x = np.min(x_values)

            regions.append((frame, BoundingBox(min_x, min_y, max_x, max_y)))

        return regions

    def contains(self, other: "TemporalContour") -> bool:
        """Whether the bounding box of this contour is greater than or equal that of another contour

        For strict containment use `other in source` vs. `source.contains(other)`.

        ## Parameters

        * `other` - another contour
        """
        lhs_region = self.bounding_box
        rhs_region = other.bounding_box

        return (
                lhs_region.left <= rhs_region.left and
                lhs_region.top <= rhs_region.top and
                lhs_region.right >= rhs_region.right and
                lhs_region.bottom >= rhs_region.bottom
        )

    def crop(self, src: np.ndarray, scaled_by: int = 1, pad_frames: FramePadding = 0,
             pad_region: PixelPadding = 0) -> np.ndarray:
        """Crops the part of a video corresponding to the contour

        Parameters
        ----------
            * `src` - source video
            * `scaled_by` - a value in {1, 2, 4, 8} indicating how much to up-scale the contour region to match the resolution of `src`
            * `pad_frames` - number of extra frames to add to start and/or end
            * `pad_region` - number of pixels to add to the bounding box dimensions
        """
        assert scaled_by in [1, 2, 4, 8], "expected scale in {1, 2, 4, 8}"

        region = self.bounding_box.scaled_up(scaled_by)
        l, t, r, b = region.left, region.top, region.right, region.bottom
        n, h, w = src.shape[:3]
        frame_start_padding, frame_end_padding = (pad_frames, pad_frames) if isinstance(pad_frames, int) else pad_frames

        if isinstance(pad_region, int):
            pixels_l, pixels_t, pixels_r, pixels_b = pad_region, pad_region, pad_region, pad_region
        elif len(pad_region) == 2:
            pixels_l, pixels_t, pixels_r, pixels_b = pad_region[0], pad_region[1], pad_region[0], pad_region[1]
        else:
            pixels_l, pixels_t, pixels_r, pixels_b = pad_region

        l = max(l - pixels_l, 0)
        t = max(t - pixels_t, 0)
        r = min(r + pixels_r, w - 1)
        b = min(b + pixels_b, h - 1)
        start_frame = max(self.frames[0] - frame_start_padding, 0)
        end_frame = min(self.frames[-1] + frame_end_padding, n - 1)
        frames = [*range(start_frame, self.frames[0]), *self.frames, *range(n, end_frame + 1)]

        return np.array([src[f][t:(b + 1), l:(r + 1), :] for f in frames])

    def density(self, digits: int = 3) -> float:
        """The ratio of points in the contour to total number of points in the region

        Parameters
        ----------
            * `digits` - number of digits to round to
        """
        return round(self.number_of_points / (len(self.frames) * self.bounding_box.width * self.bounding_box.height),
                     digits)

    def mask(self, scaled_by: int = 1) -> np.ndarray:
        """Generates a mask corresponding to the contour

        Parameters

        * `scaled_by` - a value in {1, 2, 4, 8} indicating how much to up-scale the contour region
        """
        assert scaled_by in [1, 2, 4, 8], "expected scale in {1, 2, 4, 8}"

        region = self.bounding_box.scaled_up(scaled_by)
        fmin, fmax = np.min(self.frames), np.max(self.frames)
        masks = np.zeros(((int(fmax) - fmin) + 1, region.height, region.width), dtype=np.uint8)

        for f, r, c in self.points:
            masks[f - fmin, r * scaled_by, c * scaled_by] = 255

        return masks

    def mask_from(self, src: np.ndarray, scaled_by: int = 1) -> np.ndarray:
        """Masks the portion of a video corresponding to the contour

        Parameters

        * `src` - the video to mask
        * `scaled_by` - a value in {1, 2, 4, 8} indicating how much to up-scale the contour region to match the resolution of `src`
        """
        assert scaled_by in [1, 2, 4, 8], "expected scale in {1, 2, 4, 8}"

        masks = np.zeros_like(src, dtype=np.uint8)

        for f, r, c in self.points:
            masks[f, r * scaled_by, c * scaled_by] = src[f, r * scaled_by, c * scaled_by]

        return masks

    def metadata(self):
        """A dictionary of information about this contour

        ## Metadata
        * `start_frame` - first frame in the contour
        * `nframes` - number of frames in the contour
        * `npoints` - number of points in the contour
        * `bounding_box`
            * `left`: left dimension of the box
            * `top`: top dimension of the box
            * `right`: right dimension of the box
            * `bottom` : bottom dimension of the box
            * `height` : height of the box
            * `width` : width of the box
        * `points` - list of all points in the contour
        """
        l, t, r, b = self.bounding_box.bounds

        dict(
            start_frame=self.frames[0],
            nframes=self.number_of_frames,
            npoints=self.number_of_points,
            bounding_box={
                "left": l,
                "top": t,
                "right": r,
                "bottom": b,
                "width": self.bounding_box.width,
                "height": self.bounding_box.height
            },
            points=list(map(tuple, list(self.points)))
        )

    def points_by_frame(self) -> dict[int, np.ndarray]:
        """A dictionary of points keyed by frame"""
        return {f: self.points[self.points[:, 0] == f, 1:] for f in self.frames}

    def save_mp4(self, path: str, src: np.ndarray, scaled_by: int, pad_frames: FramePadding = 0,
                 pad_region: PixelPadding = 0):
        """Crops the part of a video corresponding to the contour and saves it as an `.mp4`

        Parameters
        ----------
            * `path` - where to save the video
            * `src` - source video
            * `scaled_by` - a value in {1, 2, 4, 8} indicating how much to up-scale the contour region to match the resolution of `src`
            * `pad_frames` - number of extra frames to add to start and/or end
            * `pad_region` - number of pixels to add to the bounding box dimensions
        """
        path = Path(path)

        if Path.is_dir(path):
            raise Exception("expected file name")

        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

        cropped = self.crop(src, scaled_by, pad_frames, pad_region)

        export_to_mp4(path, cropped)

    def save_points(self, camera_id: str, date: datetime.date, time: datetime.time, bbox: BoundingBox,
                    downscaled_by: int = 1, directory: str = ""):
        """Saves the points of this contour to a numpy file

        ## Parameters

        * `camera_id` - the id of the camera used to generate the contour
        * `date` - the date of the capture
        * `time` - the time of day for the capture
        * `bbox` - the bounding box of the **full** capture
        * `downscaled_by=1` - a value in {1, 2, 4, 8} indicating how much the video was downsampled
        * `directory=""` - the directory save the file into

        The filename will be generated by joining these details. The generated filename is intended to
        be formatted such that it can be deconstructedused and used to instantiate a `BreatheCam`
        and redownload the video that produced this contour with the correct dimensions and at the correct
        resolution level.

        ## Example
        ```
        filename = `clairton4_20240519_095000_38_3000_1500_6500_2500_4.npy`
        details = filename[:-4].split("_")

        camera_id = details[0]
        day = datetime.date.strptime(details[1])
        time = datetime.time.strptime(details[2])
        nframes = int(details[3])
        bbox = BoundingBox(*map(int, details[4:8]))
        level = int(details[8])

        camera = BreatheCam(camera_id, day)
        video = camera.extract(time, nframes, bbox=bbox, seconds_per_frame=3, level=level)
        """
        filename = os.path.join(directory, self._generate_id(camera_id, date, time, bbox, downscaled_by))

        np.save(filename, self.points, allow_pickle=False)

    def __contains__(self, other: "TemporalContour") -> bool:
        """Whether the bounding box of this contour is strictly greater than that of another contour

        Use `self.contains(other)` when `self` and `other` can have equal bounding boxes.

        ## Parameters

        * `other` - another contour
        """
        lhs_region = self.bounding_box
        rhs_region = other.bounding_box

        return (
                lhs_region.left < rhs_region.left and
                lhs_region.top < rhs_region.top and
                lhs_region.right > rhs_region.right and
                lhs_region.bottom > rhs_region.bottom
        )

    def _generate_id(self, camera_id: str, date: datetime.date, time: datetime.time, bbox: BoundingBox,
                     downscaled_by: int) -> str:
        bleft, btop, bright, bbottom = bbox.bounds

        return (
            f"{camera_id}_{date.strftime('%Y%m%d')}_{time.strftime('%H%M%S')}"
            f"_{self.number_of_frames}_{bleft}_{btop}_{bright}_{bbottom}_{downscaled_by}"
        )
