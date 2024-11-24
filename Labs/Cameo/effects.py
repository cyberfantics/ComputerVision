import cv2

def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    """
    Applies edge detection with optional blur to emphasize edges.

    Parameters:
    - src: Source image.
    - dst: Destination image where the edges will be written.
    - blurKsize: Kernel size for Gaussian blur (must be odd).
    - edgeKsize: Kernel size for Sobel operator (must be odd).

    Returns:
    - None. Modifies `dst` in place.
    """
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)

def apply_portrait_effect(frame):
    """
    Applies a portrait-like effect by enhancing skin tones and adjusting contrast.

    Parameters:
    - frame: Input BGR image.

    Returns:
    - frame: Image with portrait-like effect applied.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)  # Convert to Lab color space
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)  # Improve lighting in the L-channel
    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # Optionally increase contrast for a more dramatic effect
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=5)
    return frame
