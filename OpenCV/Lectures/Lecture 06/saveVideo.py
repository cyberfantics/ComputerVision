import cv2

# Open the webcam
cam = cv2.VideoCapture(0)

# Define the codec using 'mp4v' for .mp4 files
f = cv2.VideoWriter_fourcc(*'mp4v')

# Get the resolution of the captured frames
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object to write the video
format = cv2.VideoWriter('output.mp4', f, 30, (frame_width, frame_height))

while cam.isOpened():
    r, frame = cam.read()
    if r:
        cv2.imshow("Camera", frame)
        format.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('p'):  # Press 'p' to stop recording
            break
    else:
        break

# Release everything when done
cam.release()
format.release()
cv2.destroyAllWindows()
