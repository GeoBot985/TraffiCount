from video_processing import extract_frame, read_video_metadata
from pathlib import Path

path = Path("testVid1.mp4")
meta = read_video_metadata(path)
print(meta)

frame = extract_frame(path, 0)
print("Frame:", type(frame), frame.shape if frame is not None else None)