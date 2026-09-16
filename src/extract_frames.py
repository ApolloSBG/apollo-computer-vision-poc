# Open computer vision library and load frames
import cv2

video_path = "data/raw/match_01_clip.mp4"

video = cv2.VideoCapture(video_path)

if video.isOpened():
    print("Video loaded successfully!")
else:
    print("Could not open video.")


fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frame_count / fps

print(f"Resolution: {width} x {height}")
print(f"FPS: {fps:.2f}")
print(f"Frames: {frame_count}")
print(f"Duration: {duration:.2f} seconds")

video.release()

# Extract one frame at 60 seconds
# Extract one training frame every 2 seconds
video = cv2.VideoCapture(video_path)

interval_seconds = 2
current_second = 0

while current_second < duration:
    video.set(cv2.CAP_PROP_POS_MSEC, current_second * 1000)

    success, frame = video.read()

    if success:
        filename = f"data/frames/match_01_{current_second:04d}s.jpg"
        cv2.imwrite(filename, frame)

    current_second += interval_seconds

video.release()

print("Training frames extracted successfully!")