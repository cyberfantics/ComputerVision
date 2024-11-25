import cv2
import numpy as np

video = cv2.VideoCapture(0)

kernal = np.ones((3,3), np.int8)

while True:
    r, frame = video.read()

    if r:
        frame = cv2.resize(frame, (500,500))
        frame = cv2.morphologyEx(frame, cv2.MORPH_DILATE, kernal)
        frame = cv2.Canny(frame, 50, 250)
        cv2.imshow("Camera With Morphological Operation", frame)

        if cv2.waitKey(25) & 0xff == ord('p'):
            break

    else:
        break

video.release()
cv2.destroyAllWindows()