import cv2

def returnNone(x):
    pass  # This is a placeholder for the trackbar callback


# Initialize webcam
cam = cv2.VideoCapture(0)
# Create a window
cv2.namedWindow("Effect On Images")

# Create a trackbar for effect selection
cv2.createTrackbar("Track Bar", "Effect On Images", 0, 5, returnNone)

# Check if the camera is opened correctly
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cam.read()  # Read a frame from the webcam
    if ret:
        # Get the current position of the trackbar
        bar = cv2.getTrackbarPos('Track Bar', 'Effect On Images')

        # Apply color transformations based on the trackbar value
        if bar == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif bar == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        elif bar == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif bar == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        elif bar == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the processed frame
        cv2.imshow('Effect On Images', frame)

        # Exit the loop when 'p' is pressed
        if cv2.waitKey(25) & 0xFF == ord('p'):
            break
    else:
        print("Error: Failed to capture frame.")
        break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
