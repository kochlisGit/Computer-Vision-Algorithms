import numpy as np
import time
from PIL import Image

input_filepath = 'Database/newspaper.jpg'
output_filepath = 'newspaper_binarized.jpg'

# Loads image from specified filepath.
def load_image(filepath):
    return Image.open(filepath)

# Saves image to specified filepath.
def save_image(image, filepath):
    image = image.convert('L')
    image.save(filepath)

# Converts image to array.
def image2array(image):
    return np.array(image)

# Converts an array of pixels to image.
def array2image(image_array):
    return Image.fromarray(image_array)

# Converts an RGB image to Grayscale.
# Implements weighted average values of RGB channels to make a brighter image.
def rgb2grayscale(image_array):
    return np.dot( image_array[...,:3], [0.2989, 0.5870, 0.1140] )

# Computes the integral image of an image.
def compute_integral_image(image_array, height, width):
    integral_image = np.empty_like(image_array)

    for w in range(width):
        sum = 0
        for h in range(height):
            sum += image_array[h][w]
            if w == 0:
                integral_image[h][w] = sum
            else:
                integral_image[h][w] = sum + integral_image[h][w-1]
    return integral_image

# Wellner's algorithm using SxS region of pixels and Integral Image.
def adaptive_thresholding(image_array, height, width, s, t):
    integral_image = compute_integral_image(image_array, height, width)
    output_array = np.empty_like(image_array)
    
    for w in range(width):
        for h in range(height):
            x1 = w - s
            if x1 <= 0:
                x1 = 1
            y1 = h - s
            if y1 <= 0:
                y1 = 1
            x2 = w + s
            if x2 >= width:
                x2 = width - 1
            y2 = h + s
            if y2 >= height:
                y2 = height - 1
            count = (x2 - x1)*(y2 - y1)
            sum = integral_image[y2][x2] - integral_image[y1-1][x2] - integral_image[y2][x1-1] + integral_image[y1-1][x1-1]
            if image_array[h][w] * count <= sum * ( (100-t) / 100 ):
                output_array[h][w] = 0
            else:
                output_array[h][w] = 255
    return output_array

input_image = load_image(input_filepath)
image_array = rgb2grayscale( image2array(input_image) )

height = image_array.shape[0]
width = image_array.shape[1]
s = int(width/16)
t = 15

start = time.time()
output_array = adaptive_thresholding(image_array, height, width, s, t)
end = time.time()

print('Execution time:', end - start)

output_image = array2image(output_array)
save_image(output_image, output_filepath)
