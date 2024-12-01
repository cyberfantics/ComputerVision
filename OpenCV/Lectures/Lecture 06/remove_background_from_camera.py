import cv2
import numpy as np

# Open Camera
cam = cv2.VideoCapture(0)

# Define Algorithm to remove Background with advanced options
algo1 = cv2.createBackgroundSubtractorMOG2()
kernel = np.ones((3, 3), np.uint8)  # Structuring element for morphological operations

while cam.isOpened():
    r, frame = cam.read()

    if not r:
        break

    # Resize frame
    frame_resized = cv2.resize(frame, (400, 400))

    # Apply background subtraction
    fg_mask = algo1.apply(frame_resized)

    # Apply morphological operations to clean the mask
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Fill holes

    # Use the mask to extract the foreground
    foreground = cv2.bitwise_and(frame_resized, frame_resized, mask=fg_mask)

    # Optional: Apply Gaussian blur for smoother output
    blurred_fg = cv2.GaussianBlur(foreground, (3 ,3), 0)

    # Show the resulting frame with background subtracted
    cv2.imshow("Background Subtracted Frame", blurred_fg)

    # Exit if 'p' is pressed
    if cv2.waitKey(25) & 0xff == ord('p'):
        break

# Release the camera and close windows
cam.release()
cv2.destroyAllWindows()
