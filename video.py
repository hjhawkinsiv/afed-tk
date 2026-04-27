import datetime
import ffmpeg
import json
import numpy as np
import subprocess

from collections import namedtuple


VideoInfo = namedtuple("VideoInfo", ["width", "height", "expected_frames", "start_time", "duration"])


def adjust_frame_time(time: datetime.time, seconds_per_frame: int) -> datetime.time:
    extra = time.second % seconds_per_frame

    if extra == 0:
        return time
    
    if time.second < extra:
        second = 60 + time.second - extra

        if time.minute == 0:
            if time.hour == 0:
                raise Exception(f"Invalid start frame: {time.strftime('%H:%M:%S')}")
            else:
                time = time.replace(hour=time.hour - 1, minute=59, second=second)
        else:
            time = time.replace(minute=time.minute - 1, second=second)
    else:
        time = time.replace(second=time.second - extra)

    return time


def download_video(url: str, *, start_frame=None, n_frames=None, start_time=None, end_time=None):
    width, height, expected_frames, start_time, duration = get_video_information(url, start_frame, n_frames, start_time, end_time)
    
    #Build ffmpeg command

    cmd = [
        'ffmpeg', 
        '-ss', str(start_time), 
        '-t', str(duration),
        # Add video url
        '-i', url,
        # Add output format settings
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-'
    ]

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


def export_to_mp4(path: str, video: np.ndarray):
    if not path.endswith(".mp4"):
        path = f"{path}.mp4"

    
    h, w = video.shape[1], video.shape[2]

    process = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{w}x{h}")
        .filter("pad", width="ceil(iw/2)*2", height="ceil(ih/2)*2", color="black")
        .output(path, pix_fmt="yuv420p", vcodec="libx264", r=12, loglevel="quiet")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in video:
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()


def get_video_information(url: str, start_frame=None, n_frames=None, start_time=None, end_time=None):
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
        url
    ]

    try:
        probe_output, _ = subprocess.Popen(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        metadata = json.loads(probe_output)

        if not metadata.get('streams'):
            raise ValueError("No streams found in video file")

        # Get the first video stream
        video_stream = metadata['streams'][0]

        # Extract video properties
        height, width = int(video_stream['height']), int(video_stream['width'])

        # Parse frame rate which might be in different formats
        if 'r_frame_rate' in video_stream:
            num, den = map(int, video_stream['r_frame_rate'].split('/'))
        elif 'avg_frame_rate' in video_stream:
            num, den = map(int, video_stream['avg_frame_rate'].split('/'))
        else:
            raise KeyError("Missing required video property: Could not find frame rate information")
        
        fps = num / den
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
    
    return VideoInfo(width, height, expected_frames, start_time, duration)
