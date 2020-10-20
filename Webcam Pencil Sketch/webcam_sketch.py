from cv2 import VideoCapture, waitKey , destroyAllWindows
from cv2 import imwrite, imshow
from cv2 import cvtColor, COLOR_RGB2GRAY, COLOR_GRAY2RGB
from cv2 import GaussianBlur, Canny, threshold, THRESH_BINARY_INV
from cv2 import divide

KEY_1 = 49
KEY_2 = 50
KEY_ESC = 27
KEY_ENTER = 13

# Create a color dodge, which lightens the bottom layer of the image.
def dodge(input_image, mask):
    return divide(input_image, 255 - mask, scale=256)

# 1st Filter:
# Returns a pencil sketch of the image. Uses Dodge + Gaussian Blur.
def sketch_f1(input_image):
    # 1. Convert the RGB image from webcame to Grayscale.
    gray_image = cvtColor(input_image, COLOR_RGB2GRAY)

    # 2. Invert the grayscaled image.
    inverted_gray_image = 255 - gray_image

    # 3. Apply Gaussian blur to the inverted grayscaled image.
    blurred_inverted_gray_image = GaussianBlur(inverted_gray_image, ksize=(21, 21), sigmaX=0)

    # 4. Blend the original gray image with the blurred.
    blended_image = dodge(gray_image, blurred_inverted_gray_image)

    return cvtColor(blended_image, COLOR_GRAY2RGB)

# 2nd Filter:
# Returns a pencil sketch of the image. Uses Canny Edge Detection + Gaussian Thresholding.
def sketch_f2(input_image):
    # 1. Convert the RGB image from webcame to Grayscale.
    gray_image = cvtColor(input_image, COLOR_RGB2GRAY)

    # 2. Apply Gaussian blur to the grayscaled image.
    blurred_image = GaussianBlur(gray_image, ksize=(5,5), sigmaX=0)

    # 3. Applying Canny edge detection algorithm to the blurred image.
    canny_edges = Canny(blurred_image, threshold1=10, threshold2=70)

    # 4. Applying binary thresholding.
    _, mask = threshold(canny_edges, 70, 255, THRESH_BINARY_INV)
    return mask
    
# Initializing Capture
webcam = VideoCapture(0)

# Initializing default filter.
sketch = sketch_f1

# Main Loop
key = 0
while True:
    ret, input_frame = webcam.read()
    if ret:
        webcam_sketch = sketch(input_frame)
        imshow('Webcam Sketch', webcam_sketch)
    key = waitKey(1)
    if key == KEY_1:
        sketch = sketch_f1
    elif key == KEY_2:
        sketch = sketch_f2
    elif key == KEY_ENTER:
        imwrite('sketch.jpg', webcam_sketch)
    elif key == KEY_ESC:
        break

webcam.release()
destroyAllWindows()
