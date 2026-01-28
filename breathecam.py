import datetime
import math
from itertools import product
import json
import numpy as np
import requests
import subprocess

from typing import Callable, Union

from bbox import BoundingBox
from utilities import get_frame_time

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


def _decode_video_frames(
        video_url: str,
        start_frame=None,
        n_frames=None,
        start_time=None,
        end_time=None
):
    """Downloads a video

    Parameters
    """
    # Input validation
    if (start_frame is not None) ^ ((n_frames is not None) or (end_time is not None)):
        raise ValueError("Both start_frame and n_frames must be provided together")

    if start_frame is not None and start_time is not None:
        raise ValueError("Cannot specify both frame numbers and timestamps")

    # Get video information using ffprobe
    probe_cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_format',
        '-select_streams', 'v:0',
        video_url
    ]

    try:
        probe_output, probe_error = subprocess.Popen(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        ).communicate()
        metadata = json.loads(probe_output)

        if not metadata.get('streams'):
            raise ValueError("No streams found in video file")

        # Get the first video stream
        video_stream = metadata['streams'][0]

        # Extract video properties
        try:
            width = int(video_stream['width'])
            height = int(video_stream['height'])

            # Parse frame rate which might be in different formats
            if 'r_frame_rate' in video_stream:
                num, den = map(int, video_stream['r_frame_rate'].split('/'))
                fps = num / den
            elif 'avg_frame_rate' in video_stream:
                num, den = map(int, video_stream['avg_frame_rate'].split('/'))
                fps = num / den
            else:
                raise KeyError("Could not find frame rate information")

        except KeyError as e:
            raise KeyError(f"Missing required video property: {str(e)}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe error: {e.stderr.decode()}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse FFprobe output: {str(e)}")

    # Calculate duration based on input parameters
    if start_frame is not None and n_frames is not None:
        start_time = start_frame / fps
        duration = n_frames / fps
        expected_frames = n_frames
    elif start_time is not None and end_time is not None:
        duration = end_time - start_time
        expected_frames = int(duration * fps)
    else:
        raise ValueError("Either frame numbers or timestamps must be provided")

    # Build ffmpeg command
    cmd = ['ffmpeg', '-ss', str(start_time), '-t', str(duration)]

    # Add video url
    cmd.extend(['-i', video_url])

    # Add output format settings
    cmd.extend([
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-'
    ])

    # Run ffmpeg process with communicate()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 ** 8  # Use large buffer size for video data
        )
        raw_data, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {stderr.decode()}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")

    # Verify the output size
    expected_bytes = width * height * 3 * expected_frames
    actual_bytes = len(raw_data)

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"FFmpeg output size mismatch: expected {expected_bytes} bytes "
            f"({expected_frames} frames) but got {actual_bytes} bytes "
            f"({actual_bytes // (width * height * 3)} frames)"
        )

    # Reshape into frames
    frames = np.frombuffer(raw_data, dtype=np.uint8)
    frames = frames.reshape((expected_frames, height, width, 3))

    return frames


def _get_camera_id(name: str):
    if name in CAMERAS:
        return CAMERAS[name]

    for cam in CAMERAS.values():
        if cam == name:
            return cam

    return None


class BreatheCam:
    def __init__(self,
                 location: str,
                 day: Union[datetime.date, str], *,
                 date_formatter: Union[DateFormatter, None] = None):
        if (loc_id := _get_camera_id(location)) is None:
            raise Exception(f"Invalid camera: {location}.")

        if isinstance(day, str):
            day = (date_formatter or datetime.date.fromisoformat)(day)

        self._day = day.strftime('%Y-%m-%d')
        self._root_url = f"{_TIME_MACHINES}/{loc_id}/{self._day}.timemachine"
        self._tm_url = f"{self._root_url}/tm.json"
        self._tm = requests.get(self._tm_url).json()

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
    def download(location: str,
                 day: datetime.date,
                 time: datetime.time,
                 bbox: BoundingBox,
                 frames: int = 1,
                 level: int = 1) -> np.ndarray:

        day_str = day.strftime("%Y-%m-%d")
        start_time = f"{day_str} {get_frame_time(time, 3).strftime('%H:%M:%S')}"
        camera = BreatheCam(location, day)

        if (start_frame := camera.capture_time_to_frame(start_time, 3)) < 0:
            raise Exception("First frame invalid.")

        remaining_frames = len(camera.capture_times) - start_frame

        if remaining_frames < frames:
            frames = remaining_frames

        return camera.extract(start_frame, frames, 3, bbox, level)

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
    def nlevels(self) -> str:
        return self._nlevels

    def capture_time_to_frame(self,
                              time: Union[datetime.time, str],
                              seconds_per_frame: int, *,
                              time_formatter: Union[TimeFormatter, None] = None, ) -> int:
        if isinstance(time, str):
            time = (time_formatter or datetime.time.fromisoformat)(time)

        time = get_frame_time(time, seconds_per_frame)

        return self._tm["capture-times"].index(f"{self.day} {time.strftime('%H:%M:%S')}")

    # Coordinates:  The View (rectangle) is in full-resolution coords
    # Internal to this function, the view is modified to match the subsample as the internal
    # coords are divided by subsample
    def extract(self,
                start_frame: Union[int, datetime.time],
                nframes: int,
                seconds_per_frame,
                bbox: Union[BoundingBox, None] = None,
                level: int = 1) -> np.ndarray:

        if isinstance(start_frame, datetime.time):
            time = get_frame_time(start_frame, seconds_per_frame)
            start_frame = self.capture_time_to_frame(time, seconds_per_frame)

        if start_frame < 0 or start_frame >= len(self.capture_times):
            raise Exception("First frame invalid.")

        nframes = min(nframes, len(self.capture_times) - start_frame)
        view = (bbox or BoundingBox(0, 0, self.width(), self.height())).scaled_down(level)
        level_index = self._level_index(level)
        result = np.zeros((nframes, view.height, view.width, 3), dtype=np.uint8)
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

            tile_view = BoundingBox(tile_x * tw, tile_y * th, (tile_x + 1) * tw, (tile_y + 1) * th)

            intersection = view.intersection(tile_view)

            assert intersection, f"Tile ({tile_x}, {tile_y}) does not intersect view {view}"

            src_view = intersection.translated(-tile_view.left, -tile_view.top)
            dest_view = intersection.translated(-view.left, -view.top)

            try:
                # Download the tile video
                frames = _decode_video_frames(tile_url, start_frame, nframes)

                # Copy the intersection region to the result array
                result[:, dest_view.top:dest_view.bottom, dest_view.left:dest_view.right, :] = (
                    frames[:, src_view.top:src_view.bottom, src_view.left:src_view.right, :])

            except Exception as e:
                print(f"Error processing tile {tile_url}: {str(e)}")
                continue

        return result

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
