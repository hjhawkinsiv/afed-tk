import datetime
import ffmpeg
import numpy as np
import os

from pathlib import Path
from typing import Union


FramePadding = Union[int, tuple[int, int]]
PixelPadding = Union[int, tuple[int, int], tuple[int, int, int, int]]


def _make_video_div(src: str, metadata: list[str]) -> str:
    return f"""
<div class="video-container">
  <video src="{src}" loop muted controls playbackRate=0.5></video>
  {f"<div class='metadata'>{'<br>'.join(metadata)}</div>" if metadata else ''}
</div>
"""


def _write_video_html(path: str, content: str, title: str = ""):
    html = f"""
<html>
  <head>
    <script src="js/jquery-3.7.1.min.js"></script>
    <style>
      body {{margin:0px}}
      #site-title {{
        margin-bottom: 5px;
        text-align: center;
        width: 100%;
      }}
      .label {{
        font-size: 17px;
      }}
      .video-container {{
        display: inline-flex;
        flex-flow: column nowrap;
        margin: 3px;
      }}
    </style>
  </head>
  <body>
  {f'<h1 id="site-title">{title}</h1><br><hr>' if (title := title.strip()) else ''}
  {content}

    <script>
      var ggg;
      function fixupVideo(v) {{
        console.log(v);
        var d = v.wrap('<div/>');
        console.log(d);
        if (v.attr('playbackRate')) {{
          v[0].playbackRate = parseFloat($(v).attr('playbackRate'));
        }}
        if (v.attr('trimRight')) {{
          vvv = v;
          ddd = d;
        }}
      }}

      function init() {{
        console.log('init');
        let observer = new IntersectionObserver(
          (entries, observer) => {{
            for (entry of entries) {{
              if (entry.isIntersecting) {{
            console.log('play', entry.target);
                entry.target.play();
              }} else  {{
          console.log('pause', entry.target);
                entry.target.pause(); 
        }}
            }}
          }},
          {{threshold: 0}}
        );

        $('img,video').each(function(i,v){{
          fixupVideo($(v));
          console.log('setting up', v);
          observer.observe(v);
        }});
      }}

      $(init);
    </script>
  </body>
</html>
"""
    with open(path, "w") as htmlFile:
        htmlFile.write(html)


def backtrack(time: datetime.time, seconds: int) -> datetime.time:
    """Steps a time instance back a number of seconds

    Parameters
    ----------
    time - the time
    seconds - the number of seconds to step back
    """
    if time.second < seconds:
        second = 60 + time.second - seconds

        if time.minute == 0:
            if time.hour == 0:
                raise Exception(f"Invalid start frame: {time.strftime('%H:%M:%S')}")
            else:
                time = time.replace(hour=time.hour - 1, minute=59, second=second)
        else:
            time = time.replace(minute=time.minute - 1, second=second)
    else:
        time = time.replace(second=time.second - seconds)

    return time


def export_contours_to_html(html_path: str,
                            videos_dir: str,
                            contours,
                            src: np.ndarray,
                            src_nlevels: int,
                            contour_nlevels: int,
                            pad_frames: FramePadding = 0,
                            pad_region: PixelPadding = 0,
                            title: str = "",
                            video_file_prefix: str = "event"):
    divs = []
    videos_dir = videos_dir.strip()

    if videos_dir.endswith('/'):
        videos_dir = videos_dir[:-1]

    Path(os.path.dirname(html_path)).mkdir(parents=True, exist_ok=True)
    Path(videos_dir).mkdir(parents=True, exist_ok=True)

    for i, contour in enumerate(contours):
        video_name = f"{videos_dir}/{video_file_prefix}{i + 1}.mp4"

        try:
            export_to_mp4(video_name, contour.crop(src, src_nlevels, contour_nlevels, pad_frames, pad_region))
            f, h, w = contour.number_of_frames, contour.height, contour.width
            region = contour.region.upsample(src_nlevels)

            metadata = [
                f"Video {i + 1}",
                f"Start Frame: {contour.frames[0]}; ",
                f"Dimensions: {f}x{h}x{w}; ",
                f"Points in Contour: {contour.number_of_points}; ",
                f"Contour Area: {contour.width * contour.height}; ",
                f"Point Density: {contour.density(5)}; ",
                f"View: ({region.left}, {region.top}, {region.right}, {region.bottom})"
            ]

            divs.append(_make_video_div(video_name, metadata))
        except BrokenPipeError as e:
            print(e)

    _write_video_html(html_path, '\n'.join(divs), title)


def export_to_mp4(path: Union[str, os.PathLike], video: np.ndarray):
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

def get_frame_time(time: datetime.time, seconds_per_frame: int) -> datetime.time:
    extra = time.second % seconds_per_frame

    return time if extra == 0 else backtrack(time, extra)


def get_previous_frame_time(time: datetime.time, seconds_per_frame: int, nframes: int = 1) -> datetime.time:
    extra = time.second % seconds_per_frame

    return backtrack(time, seconds_per_frame * nframes if extra == 0 else (seconds_per_frame - 1) * nframes + extra)
