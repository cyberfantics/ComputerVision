import cv2

cam = cv2.VideoCapture(0)

# Mask To Subtract Background From Live Video
mask = cv2.createBackgroundSubtractorKNN()
while True:
    r, frame = cam.read()

    if r==True:
        frame = cv2.resize(frame, (500,500))
        frame = mask.apply(frame)
        cv2.imshow("Video", frame)

        if cv2.waitKey(25) & 0xff == ord('p'):
            break
    else:
        break

cam.release()
cv2.destroyAllWindows()