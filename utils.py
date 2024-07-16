import imageio
import os
from typing import Iterable
import numpy as np

def render(env):
    """Renders the current environment state to an image."""
    return env.physics.render(camera_id=0, width=640, height=480)

def write_video(
    filepath: os.PathLike,
    frames: Iterable[np.ndarray],
    fps: int = 60,
    quality: int = 10,
    verbose: bool = False,
    **kwargs
):
    """
    Saves a sequence of frames as a video file.

    Parameters:
    - filepath: Path to save the video file.
    - frames: An iterable of frames (numpy arrays).
    - fps: Frames per second (default: 60).
    - quality: Output video quality (default: 10).
    - verbose: If True, prints the save location.
    - **kwargs: Additional arguments for imageio.get_writer.

    Returns: None
    """

    writer = imageio.get_writer(filepath, fps=fps, quality=quality, format='FFMPEG', **kwargs)
    
    try:
        if verbose:
            print(f'Saving video to: {filepath}')
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    if verbose:
        print(f'Video saved to: {filepath}')

