import datetime
import math
import numpy as np
import requests
import warnings

from collections.abc import Callable
from itertools import product

from boundaries import BoundingRect
from video import adjust_frame_time, download_video

DateFormatter = Callable[[str], datetime.date]
TimeFormatter = Callable[[str], datetime.time]

_TIME_MACHINES = "https://tiles.cmucreatelab.org/ecam/timemachines"

CAMERAS = {
    "Clairton Coke Works": "clairton4",
    "Shell Plastics West": "vanport3",
    "Edgar Thomson South": "westmifflin2",
    "Metalico": "accan2",
    "Revolution ETC/Harmon Creek Gas Processing Plants": "cryocam",
    "Riverside Concrete": "cementcam",
    "Shell Plastics East": "center1",
    "Irvin": "irvin1",
    "North Shore": "heinz",
    "Mon. Valley": "walnuttowers1",
    "Downtown": "trimont1",
    "Oakland": "oakland"
}


def breathecam_id(name: str):
    if name in CAMERAS:
        return CAMERAS[name]

    for cam in CAMERAS.values():
        if cam == name:
            return cam

    return None


class BreatheCam:
    def __init__(self, location: str, day: datetime.date | str, *, date_formatter: DateFormatter | None = None):
        if (loc_id := breathecam_id(location)) is None:
            loc_id = location
            warnings.warn(f"Unrecognized camera id: {location}.")

        if isinstance(day, str):
            day = (date_formatter or datetime.date.fromisoformat)(day)

        self._location = location
        self._location_id = loc_id
        self._day = day.strftime('%Y-%m-%d')
        self._root_url = f"{_TIME_MACHINES}/{loc_id}/{self._day}.timemachine"
        self._tm_url = f"{self._root_url}/tm.json"
        self._tm = requests.get(self._tm_url).json()

        capture_times = self._tm["capture-times"]
        time_x = datetime.datetime.fromisoformat(capture_times[0])
        time_y = datetime.datetime.fromisoformat(capture_times[1])
        self._seconds_per_frame = int((time_y - time_x).total_seconds())

        datasets = self._tm["datasets"]

        assert len(datasets) == 1

        dataset = datasets[0]
        dataset_id = dataset["id"]

        self._tile_root_url = f"{self._root_url}/{dataset_id}"
        self._r_url = f"{self._tile_root_url}/r.json"
        self._r = requests.get(self._r_url).json()
        self._nlevels = self._r["nlevels"]
        self._tile_height = self._r["video_height"]
        self._tile_width = self._r["video_width"]


    @staticmethod
    def download(location: str, day: datetime.date, time: datetime.time, view: BoundingRect, n_frames: int = 1, level: int = 1) -> np.ndarray:
        day_str = day.strftime("%Y-%m-%d")
        camera = BreatheCam(location, day)
        start_time = f"{day_str} {adjust_frame_time(time, camera.seconds_per_frame).strftime('%H:%M:%S')}"
        
        if (start_frame := camera.capture_time_to_frame(start_time)) < 0:
            raise Exception("First frame invalid.")

        remaining_frames = len(camera.capture_times) - start_frame

        if remaining_frames < n_frames:
            n_frames = remaining_frames

        return camera.extract(start_frame, n_frames, view, level)

    @property
    def capture_times(self) -> list[str]:
        return self._tm["capture-times"]

    @property
    def day(self) -> str:
        return self._day

    @property
    def fps(self) -> float:
        return self._r["fps"]

    @property
    def location(self) -> str:
        return self._location

    @property
    def location_id(self) -> str:
        return self._location_id

    @property
    def nlevels(self) -> str:
        return self._nlevels

    @property
    def seconds_per_frame(self) -> int:
        return self._seconds_per_frame

    def capture_time_to_frame(self, time: datetime.time | str, *, time_formatter: TimeFormatter | None = None) -> int:
        if isinstance(time, str):
            time = (time_formatter or datetime.time.fromisoformat)(time)

        time = adjust_frame_time(time, self._seconds_per_frame)

        return self._tm["capture-times"].index(f"{self.day} {time.strftime('%H:%M:%S')}")

    # Coordinates:  The View (rectangle) is in full-resolution coords
    # Internal to this function, the view is modified to match the subsample as the internal
    # coords are divided by subsample
    def extract(self,frame: int | datetime.time, n_frames: int, view: BoundingRect | None = None, subsample_factor: int = 1) -> np.ndarray:
        if isinstance(frame, datetime.time):
            time = adjust_frame_time(frame, self._seconds_per_frame)
            frame = self.capture_time_to_frame(time)

        if frame < 0 or frame >= len(self.capture_times):
            raise Exception("First frame invalid.")

        n_frames = min(n_frames, len(self.capture_times) - frame)
        view = (view or BoundingRect(0, 0, self.width(), self.height())).contracted_by(subsample_factor)
        level_index = self._level_index(subsample_factor)
        result = np.zeros((n_frames, view.height, view.width, 3), dtype=np.uint8)
        th, tw = self._tile_height, self._tile_width
        min_tile_y = view.top // th
        max_tile_y = 1 + (view.bottom - 1) // th
        min_tile_x = view.left // tw
        max_tile_x = 1 + (view.right - 1) // tw

        for tile_y, tile_x in product(range(min_tile_y, max_tile_y), range(min_tile_x, max_tile_x)):
            tile_url = self._tile_url(level_index, tile_x, tile_y)
            response = requests.head(tile_url)

            if response.status_code == 404:
                print(f"Warning: tile {tile_x},{tile_y} does not exist, skipping...")
                continue

            tile_view = BoundingRect(tile_x * tw, tile_y * th, (tile_x + 1) * tw, (tile_y + 1) * th)
            intersection = view.intersection(tile_view)

            assert intersection, f"Tile ({tile_x}, {tile_y}) does not intersect view {view}"

            src_view = intersection.translated(-tile_view.left, -tile_view.top)
            dest_view = intersection.translated(-view.left, -view.top)
            sl, st, sr, sb = src_view.bounds
            dl, dt, dr, db = dest_view.bounds

            try:
                # Download the tile video
                frames = download_video(tile_url, start_frame=frame, n_frames=n_frames)

                # Copy the intersection region to the result array
                result[:, dt:db, dl:dr, :] = frames[:, st:sb, sl:sr, :]

            except Exception as e:
                print(f"Error processing tile {tile_url}: {str(e)}")
                continue

        return result

    def frame_to_capture_time(self, frame: int) -> datetime.datetime | None:
        capture_times = self._tm["capture-times"]

        if frame < 0 or frame >= len(capture_times):
            return None
        
        day_str, time_str = capture_times[frame].split(" ")
        day = datetime.date.fromisoformat(day_str)
        time = datetime.time.fromisoformat(time_str)

        return datetime.datetime.combine(day, time)


    def height(self, level: int = 1) -> int:
        return int(math.ceil(self._r["height"] / level))

    def width(self, level: int = 1) -> int:
        return int(math.ceil(self._r["width"] / level))

    def _level_index(self, level: int) -> int:
        assert ((level & (level - 1)) == 0), "Expected value in {1, 2, 4, 8}"

        index = self._nlevels - level.bit_length()

        assert index >= 0, f"Sample level {level} is invalid for timemachine with {self._nlevels} levels."

        return index

    def _tile_url(self, level: int, tile_x: int, tile_y: int) -> str:
        return f"{self._tile_root_url}/{level}/{tile_y * 4}/{tile_x * 4}.mp4"


class BreatheCamCapture:
    __slots__ = [
        "view",
        "playback_speed"
        "begin_time",
        "end_time",
        "start_dwell",
        "end_dwell",
        "day",
        "location",
        "format",
        "start_frame",
        "fps",
        "width",
        "height",
        "tile_format",
        "from_screenshot",
        "minimal_ui",
        "watermark"
    ]

    def __init__(
        self,
        location: str,
        view: BoundingRect,
        begin_time: str,
        end_time: str,
        start_frame: int,
        fps: int,
        width: int,
        height: int
    ):
        self.view = view
        self.playback_speed = 50
        self.begin_time = begin_time
        self.end_time = end_time
        self.start_dwell = 0
        self.end_dwell = 0
    
        bt_date = datetime.datetime.fromisoformat(begin_time)

        self.day = bt_date.date().strftime("%Y-%m-%d")
        self.location = location
        self.format = "mp4"
        self.start_frame = start_frame
        self.fps = fps
        self.width = width
        self.height = height
        self.tile_format = "mp4"
        self.from_screenshot = True
        self.minimal_ui = True
        self.watermark = "Breathe%20Project%7CCREATE%20Lab"

    def generate_thumbnail_link(self) -> str:
        return (f"https://thumbnails-v2.createlab.org/thumbnail?root=https%3A%2F%2Fbreathecam.org%23"
                f"v%3D{self.view.left},{self.view.top},{self.view.right},{self.view.bottom},pts"
                f"%26ps%3D{self.playback_speed}"
                f"%26bt%3D{self.begin_time}"
                f"%26et%3D{self.end_time}"
                f"%26startDwell%3D{self.start_dwell}"
                f"%26endDwell%3D{self.end_dwell}"
                f"%26d%3D{self.day}"
                f"%26s%3D{self.location}"
                f"%26fps%3D{self.fps}"
                f"%26width%3D{self.width}"
                f"%26height%3D{self.height}"
                f"%26format%3D{self.format}"
                f"%26tileFormat%3D{self.tile_format}"
                f"{'%26fromScreenshot' if self.from_screenshot else ''}"
                f"{'%26minimalUI' if self.minimal_ui else '' }"
                f"%26watermark%3D{self.watermark}")
    
    def generate_share_link(self) -> str:
        return (f"https://breathecam.org/#"
                f"v={self.view.left},{self.view.top},{self.view.right},{self.view.bottom},pts"
                f"&t={float(self.start_frame) / self.fps}"
                f"&ps={self.playback_speed}"
                f"&bt={self.begin_time}"
                f"&et={self.end_time}"
                f"&d={self.day}"
                f"&s={self.location}")

