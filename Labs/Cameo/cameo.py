import cv2
from managers import WindowManager, CaptureManager
from effects import strokeEdges, apply_portrait_effect

class Cameo:
    def __init__(self):
        self._windowManager = WindowManager("Cameo", keypressCallback=self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)

    def run(self):
        self._windowManager.createWindow()

        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            if frame is not None:
                processed_frame = self.process_frame(frame)
                self._captureManager._frame = processed_frame

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

        self._captureManager._capture.release()

    def process_frame(self, frame):
        """
        Processes the frame by applying both stroke edges and portrait-like effect.
        """
        strokeEdges(frame, frame, blurKsize=5, edgeKsize=3)  # Apply strokeEdges effect in-place
        frame = apply_portrait_effect(frame)  # Apply portrait-like effect
        return frame

    def onKeypress(self, keycode):
        if keycode == 27:  # Escape key
            self._windowManager.destroyWindow()
        elif keycode == ord('s'):  # 's' key to save an image
            self._captureManager.writeImage("capture.jpg")
        elif keycode == ord('v'):  # 'v' key to start video recording
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo("output.mp4")
        elif keycode == ord('b'):  # 'b' key to stop video recording
            if self._captureManager.isWritingVideo:
                self._captureManager.stopWritingVideo()


if __name__ == "__main__":
    Cameo().run()
