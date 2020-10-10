import cv2
import numpy as np

ESC = 27
ENTER = 13

# Create a color dodge, which lightens the bottom layer of the image.
def dodge(input_image, mask):
    return cv2.divide(input_image, 255 - mask, scale=256)

# Returns a sketch of the image. Uses Gaussian Thresholding.
def sketch(input_image):
    # 1. Convert the RGB image from webcame to Grayscale.
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    # 2. Invert the grayscaled image.
    inverted_gray_image = 255 - gray_image

    # 3. Apply Gaussian blur to the inverted grayscaled image.
    blurred_inverted_gray_image = cv2.GaussianBlur(inverted_gray_image, ksize=(21, 21), sigmaX=0)

    # 4. Blend the original gray image with the blurred.
    blended_image = dodge(gray_image, blurred_inverted_gray_image)

    return cv2.cvtColor(blended_image, cv2.COLOR_GRAY2RGB)

# Initializing Capture
webcam = cv2.VideoCapture(0)

# Main Loop
key = 0
while True:
    ret, input_frame = webcam.read()
    if ret:
        webcam_sketch = sketch(input_frame)
        cv2.imshow('Webcam Sketch', webcam_sketch)
    key = cv2.waitKey(1)
    if key == ENTER:
        cv2.imwrite('sketch.jpg', webcam_sketch)
    if key == ESC:
        break

webcam.release()
cv2.destroyAllWindows()