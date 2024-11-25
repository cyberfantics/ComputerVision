import cv2

cam = cv2.VideoCapture(0)

while True:
    r, frame = cam.read()

    if r == True:
        frame = cv2.resize(frame, (300,300))
        cv2.imshow('Camera', frame)

        if cv2.waitKey(25) & 0xff == ord('p'):
            break
    
    else:
        break

cam.release()
cv2.destroyAllWindows()