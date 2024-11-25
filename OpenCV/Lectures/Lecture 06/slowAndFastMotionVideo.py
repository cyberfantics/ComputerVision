import cv2, sys

# Manually specify the path to the project directory
project_path = '../../../'

# Add the project directory to sys.path
sys.path.append(project_path)

from mansoor import Repeatable

r = Repeatable(300, 300, base_path='../../../data/images/', video_path='../../../data/video/')  # It has functions that we need to do Repeatedly

video = r.load_video()

# Get the video's frames per second (fps)
fps = video.get(cv2.CAP_PROP_FPS)

# Set initial delay based on fps (this will help maintain the video’s normal speed)
delay = int(1000 / fps)  # Delay in milliseconds per frame

while video.isOpened():
    ret, frame = video.read()

    if ret:
        cv2.imshow('Video', frame)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord('f'):  # Fast motion
            # Decrease delay to speed up playback
            delay = max(1, delay // 2)  # Min delay is 1ms
            print(f'Currently Video is playing at faster speed with delay {delay}ms')

        elif key == ord('s'):  # Slow motion
            # Increase delay to slow down playback
            delay = max(1, delay * 2)
            print(f'Currently Video is playing at slower speed with delay {delay}ms')

        elif key == ord('q'):  # Quit video
            print("Breaking Video")
            break
    else:
        break

video.release()
cv2.destroyAllWindows()
