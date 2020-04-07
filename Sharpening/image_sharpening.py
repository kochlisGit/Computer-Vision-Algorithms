import cv2 as cv
import numpy as np

input_filepath = 'Database/Cat.jpg'
output1_filepath = 'cat_sharpened_filter.jpg'
output2_filepath = 'cat_sharpened_unmask.jpg'

# Sharpens the image using Laplacian Filter.
def sharpen_image_with_kernel(image):
    sharpen_kernel = np.array( [ [0, -1, 0], [-1, 5, -1], [0, -1, 0] ] )
    return cv.filter2D(image, -1, sharpen_kernel)

# Sharpens the image using Unsharp Mark.
# Unsharp Mark is more robust to noise, because It first removes noise.
# Also, You can control the amount of sharpness.
def sharpen_image_with_unsharp_mask(image, kernel_size = (5, 5), sigma = 1.0, amount = 1.0, threshold = 0):
    blurred_image = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened_image = float(amount + 1) * image - float(amount) * blurred_image
    sharpened_image = np.maximum( sharpened_image, np.zeros(sharpened_image.shape) )
    sharpened_image = np.minimum( sharpened_image, 255 * np.ones(sharpened_image.shape) )
    sharpened_image = sharpened_image.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred_image) < threshold
        np.copyto(sharpened_image, image, where = low_contrast_mask)
    return sharpened_image

image = cv.imread(input_filepath)

sharpened_image_kernel = sharpen_image_with_kernel(image)
sharpened_image_unsharp = sharpen_image_with_unsharp_mask(image)

cv.imwrite(output1_filepath, sharpened_image_kernel)
cv.imwrite(output2_filepath, sharpened_image_unsharp)
