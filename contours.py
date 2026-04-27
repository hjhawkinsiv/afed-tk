import numpy as np
import os

from math import floor
from pathlib import Path
from typing import Union

from boundaries import BoundingRect, BoundingCircle
from video import export_to_mp4

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
        self._min_bbox = None
        self._min_bcir = None

    @staticmethod
    def load_points(contour_points_file):
        """Creates a `TemporanContour` instance defined by points saved using `np.save` with `allow_pickle` set to `False`

        ## Parameters

        * `contour_points_file:` - source of the points

        """
        return TemporalContour(np.load(contour_points_file, allow_pickle=False))

    @property
    def height(self) -> int:
        """The height of the minimum bounding box."""
        return self.minimum_bounding_box().height

    @property
    def width(self) -> int:
        """The width of the minimum bounding box."""
        return self.minimum_bounding_box().width
    
    def crop(
            self, 
            src: np.ndarray, 
            upsample_factor: int = 1, 
            pad_frames: FramePadding = 0,
            pad_region: PixelPadding = 0) -> np.ndarray:
        """Crops the part of a video corresponding to the contour

        Parameters
        ----------
            * `src` - source video
            * `upsample_factor` - a value in {1, 2, 4, 8} indicating how much to up-scale the contour region to match the resolution of `src`
            * `pad_frames` - number of extra frames to add to start and/or end
            * `pad_region` - number of pixels to add to the bounding box dimensions
        """
        assert upsample_factor in [1, 2, 4, 8], "expected scale in {1, 2, 4, 8}"

        region = self.minimum_bounding_box.dilated_by(upsample_factor)
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
        return round(self.number_of_points / (len(self.frames) * self.minimum_bounding_box.width * self.minimum_bounding_box.height),
                     digits)

    def mask(self, upsample_factor: int = 1) -> np.ndarray:
        """Generates a mask corresponding to the contour

        Parameters

        * `upsample_factor` - a value in {1, 2, 4, 8} indicating how much to up-scale the contour region
        """
        assert upsample_factor in [1, 2, 4, 8], "expected scale in {1, 2, 4, 8}"

        region = self.minimum_bounding_box.dilated_by(upsample_factor)
        fmin, fmax = np.min(self.frames), np.max(self.frames)
        masks = np.zeros(((int(fmax) - fmin) + 1, region.height, region.width), dtype=np.uint8)

        for f, r, c in self.points:
            masks[f - fmin, r * upsample_factor, c * upsample_factor] = 255

        return masks

    def mask_from(self, src: np.ndarray, upsample_factor: int = 1) -> np.ndarray:
        """Masks the portion of a video corresponding to the contour

        Parameters

        * `src` - the video to mask
        * `upsample_factor` - a value in {1, 2, 4, 8} indicating how much to up-scale the contour region to match the resolution of `src`
        """
        assert upsample_factor in [1, 2, 4, 8], "expected scale in {1, 2, 4, 8}"

        masks = np.zeros_like(src, dtype=np.uint8)

        for f, r, c in self.points:
            masks[f, r * upsample_factor, c * upsample_factor] = src[f, r * upsample_factor, c * upsample_factor]

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
        min_bbox = self.minimum_bounding_box()
        l, t, r, b = min_bbox.bounds

        dict(
            start_frame=self.frames[0],
            nframes=self.number_of_frames,
            npoints=self.number_of_points,
            bounding_box={
                "left": l,
                "top": t,
                "right": r,
                "bottom": b,
                "width": min_bbox.width,
                "height": min_bbox.height
            },
            points=list(map(tuple, list(self.points)))
        )

    def minimum_bounding_box(self) -> BoundingRect:
        """The smallest bounding box containing every point in the contour"""
        if self._min_bbox is None:
            y, x = self.points[:, 1], self.points[:, 2]
            left, top, right, bottom = np.min(x), np.min(y), np.max(x), np.max(y)
            self._min_bbox = BoundingRect(left, top, right, bottom)

        return self._min_bbox

    def minimum_bounding_circle(self) -> BoundingCircle:
        """The smallest bounding box containing every point in the contour"""
        min_bbox = self.minimum_bounding_box()
        centroid_x, centroid_y = 0.0, 0.0

        for frame in self.frames:
            mask_candidates = self.points[:, 0] == frame
            points = self.points[mask_candidates, 1:]

            m00 = len(points)
            m10 = np.sum(points[:, 1])
            m01 = np.sum(points[:, 0])
            centroid_x += m10 / m00
            centroid_y += m01 / m00

        centroid_x = floor(centroid_x / len(self.frames))
        centroid_y = floor(centroid_y / len(self.frames))

        l_radius = abs(centroid_x - min_bbox.left)
        t_radius = abs(centroid_y - min_bbox.top)
        r_radius = abs(min_bbox.right - centroid_x)
        b_radius = abs(min_bbox.bottom - centroid_y)

        return BoundingCircle(centroid_x, centroid_y, max(l_radius, t_radius, r_radius, b_radius))

    def points_by_frame(self) -> dict[int, np.ndarray]:
        """A dictionary of points keyed by frame"""
        return {f: self.points[self.points[:, 0] == f, 1:] for f in self.frames}

    def save_mp4(
            self, 
            path: str, 
            src: np.ndarray, 
            upsample_factor: int, 
            pad_frames: FramePadding = 0,
            pad_region: PixelPadding = 0
    ):
        """Crops the part of a video corresponding to the contour and saves it as an `.mp4`

        Parameters
        ----------
            * `path` - where to save the video
            * `src` - source video
            * `upsample_factor` - a value in {1, 2, 4, 8} indicating how much to up-scale the contour region to match the resolution of `src`
            * `pad_frames` - number of extra frames to add to start and/or end
            * `pad_region` - number of pixels to add to the bounding box dimensions
        """
        path = Path(path)

        if Path.is_dir(path):
            raise Exception("expected file name")

        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

        cropped = self.crop(src, upsample_factor, pad_frames, pad_region)

        export_to_mp4(path, cropped)
