import matplotlib.pyplot as plt
import numpy as np
import cv2, sys


# Manually specify the path to the project directory
project_path = '../../../'

# Add the project directory to sys.path
sys.path.append(project_path)

from mansoor import Repeatable

r = Repeatable(300,300, base_path='../../../data/images/', video_path='../../../data/video/') # It have functions that we need to do Repeatedly

video = r.load_video()

while video.isOpened():
    r, frame = video.read()
    if r == True:
        cv2.imshow("Video", frame)

        if cv2.waitKey(30) & 0xff == ord('p'):
            break

    else:
        break