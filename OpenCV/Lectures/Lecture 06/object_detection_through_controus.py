import cv2
import numpy as np

# Open the webcam
cam = cv2.VideoCapture(0)

def mansoor(x):
    pass

# Create a window for the camera feed
cv2.namedWindow("Camera")

# TrackBar for threshold
cv2.createTrackbar('th', 'Camera', 0, 255, mansoor)  # Threshold

# Create TrackBars for Lower Bound (HSV)
cv2.createTrackbar('lb', 'Camera', 0, 255, mansoor)  # Lower bound - Blue
cv2.createTrackbar('lg', 'Camera', 0, 255, mansoor)  # Lower bound - Green
cv2.createTrackbar('lr', 'Camera', 0, 255, mansoor)  # Lower bound - Red

# Create TrackBars for Upper Bound (HSV)
cv2.createTrackbar('ub', 'Camera', 255, 255, mansoor)  # Upper bound - Blue
cv2.createTrackbar('ug', 'Camera', 255, 255, mansoor)  # Upper bound - Green
cv2.createTrackbar('ur', 'Camera', 255, 255, mansoor)  # Upper bound - Red

while cam.isOpened():
    ret, frame = cam.read()

    if not ret:
        break

    thr = cv2.getTrackbarPos('th', 'Camera')
    # Get the trackbar positions for the lower and upper bounds
    lb = cv2.getTrackbarPos('lb', "Camera")
    lg = cv2.getTrackbarPos('lg', "Camera")
    lr = cv2.getTrackbarPos('lr', "Camera")

    ub = cv2.getTrackbarPos('ub', "Camera")
    ug = cv2.getTrackbarPos('ug', "Camera")
    ur = cv2.getTrackbarPos('ur', "Camera")

    # Convert the trackbar values to numpy arrays
    low = np.array([lb, lg, lr])
    high = np.array([ub, ug, ur])
    
    # Resize Frame To Get Smaller Display
    frame_resized = cv2.resize(frame, (400, 400))

    # Convert the frame to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # Create a mask based on the lower and upper bounds
    mask = cv2.inRange(hsv, low, high)

    # Bitwise AND the original frame and the mask to get the filtered image
    filtered_image = cv2.bitwise_and(frame_resized, frame_resized, mask=mask)

    # Find Threshold
    _, t = cv2.threshold(mask, thr, 255, cv2.THRESH_BINARY)

    # Find contours
    cnt, _ = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw Contours
    cv2.drawContours(frame_resized, cnt, -1, (255, 0, 255), 2)

    # Display the original frame
    cv2.imshow('Original Camera Feed', frame_resized)

    # Display the mask (the part that matches the color)
    cv2.imshow('Mask', mask)
# Display the mask (the part that matches the color)
    cv2.imshow('Camera', mask)

    # Display the HSV image (this is the converted BGR image to HSV)
    cv2.imshow('HSV Image', hsv)

    # Display the filtered image
    cv2.imshow('Filtered Camera Feed', filtered_image)

    # Show current trackbar values on the frame (for real-time feedback)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_resized, f'Lower Bound: B={lb}, G={lg}, R={lr}', (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_resized, f'Upper Bound: B={ub}, G={ug}, R={ur}', (10, 60), font, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_resized, f'Threshold: {thr}', (10, 90), font, 0.8, (255, 255, 255), 2)

    # Wait for a key press, and if 'p' is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
