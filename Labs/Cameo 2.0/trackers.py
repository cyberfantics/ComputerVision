import cv2
import rects
import utils


class FaceTracker(object):
    """A tracker for facial features: face, eyes, nose, mouth."""
    
    def __init__(self, scaleFactor=1.2, minNeighbors=2):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        
        self._faces = []
        
        self._faceClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_frontalface_alt.xml')
        self._eyeClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_eye.xml')
        self._noseClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_nose.xml')
        self._mouthClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_mouth.xml')

    @property
    def faces(self):
        """The tracked facial features."""
        return self._faces
    
    def update(self, image):
        """Update the tracked facial features."""
        
        self._faces = []
        
        if utils.isGray(image):
            image = cv2.equalizeHist(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(image, image)
        
        minSize = utils.widthHeightDividedBy(image, 8)
        
        faceRects = self._faceClassifier.detectMultiScale(
            image, self.scaleFactor, self.minNeighbors, minSize=minSize)
        
        if faceRects is not None:
            for faceRect in faceRects:
                face = Face()
                face.faceRect = faceRect
                
                x, y, w, h = faceRect
                
                # Seek an eye in the upper-left part of the face.
                searchRect = (x + w // 7, y, w * 2 // 7, h // 2)
                face.leftEyeRect = self._detectOneObject(
                    self._eyeClassifier, image, searchRect, 64)
                
                # Seek an eye in the upper-right part of the face.
                searchRect = (x + w * 4 // 7, y, w * 2 // 7, h // 2)
                face.rightEyeRect = self._detectOneObject(
                    self._eyeClassifier, image, searchRect, 64)
                
                # Seek a nose in the middle part of the face.
                searchRect = (x + w // 4, y + h // 4, w // 2, h // 2)
                face.noseRect = self._detectOneObject(
                    self._noseClassifier, image, searchRect, 32)
                
                # Seek a mouth in the lower-middle part of the face.
                searchRect = (x + w // 6, y + h * 2 // 3, w * 2 // 3, h // 3)
                face.mouthRect = self._detectOneObject(
                    self._mouthClassifier, image, searchRect, 16)
                
                self._faces.append(face)
