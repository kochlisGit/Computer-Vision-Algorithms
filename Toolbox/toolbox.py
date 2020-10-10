import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Converts RBG to Grayscale image.
# Params: None
def rgb2gray( input_image, params=None ):
    return cv2.cvtColor(src=input_image, code=cv2.COLOR_BGR2GRAY)

# Constructs the histogram of the image channels.
# Params: r|g|b.
def hist( input_image, params=['r', 'g', 'b'] ):
    for n, color in enumerate(params):
        hist = cv2.calcHist( [input_image], [n], None, [256], [0, 256] )
        plt.plot(hist, color=color)
        plt.xlim( [0, 256] )
    plt.savefig('plot.jpg')
    return cv2.imread('plot.jpg')

# Draws a line or a rectangle in the image.
# Params: line/rectangle x1 y1 x2 y2 r g b thickness
def draw(input_image, params):
    shape = params[0]
    x1, y1, x2, y2, r, g, b, thick = [ int(param) for param in params[1:] ]
    if shape == 'line':
        cv2.line(input_image, pt1=(x1, y1), pt2=(x2, y2), color=(r,g,b), thickness=thick)
        return input_image
    elif shape == 'rectangle':
        cv2.rectangle(input_image, pt1=(x1, y1), pt2=(x2, y2), color=(r,g,b), thickness=thick)
        return input_image
    else:
        return np.zeros(input_image.shape)

# Translates image using a Translation Matrix.
# Params: Dx Dy.
def translate(input_image, params):
    height, width = input_image.shape[:2]
    dx, dy = [int(param) for param in params]
    T = np.float32( [
        [1, 0, dx],
        [0, 1, dy]
    ] )
    return cv2.warpAffine(input_image, T, (width, height) )

# Rotate image using a Rotation Matrix.
# Params: Cx Cy angle
def rotate(input_image, params):
    height, width = input_image.shape[:2]
    center_x, center_y, a = [int(param) for param in params]
    R = cv2.getRotationMatrix2D(center=(center_x, center_y), angle=a, scale=1)
    return cv2.warpAffine(input_image, R, (width, height) )

# Scale pixels using a Scaling Matrix.
# Params: Sx, Sy
def resize(input_image, params):
    Sx, Sy = [float(param) for param in params]
    return cv2.resize(input_image, dsize=None, fx=Sx, fy=Sy)

# Crops a region of the image.
# Params: x1 x2 y1 y2
def crop(input_image, params):
    x1, y1, x2, y2 = [int(param) for param in params]
    return input_image[y1:y2, x1:x2]

# Brightens the entire image.
# Params: brightness
def brighten(input_image, params):
    brightness = int( params[0] )
    if brightness > 255:
        brightness = 255
    M = np.ones(shape=input_image.shape, dtype=np.uint8) * brightness
    return cv2.add(input_image, M)

# Darkens the entire image.
# Params: darkness
def darken(input_image, params):
    darkness = int( params[0] )
    if darkness > 255:
        darkness = 255
    M = np.ones(shape=input_image.shape, dtype=np.uint8) * darkness
    return cv2.subtract(input_image, M)

# Blurs the entire image.
# Params: average/gaussian/median/bilateral kernel_size
def blur(input_image, params):
    blurring = params[0]
    kernel_size = int( params[1] )
    kernel = (kernel_size, kernel_size)
    if blurring == 'average':
        return cv2.blur(input_image, kernel)
    elif blurring == 'gaussian':
        return cv2.GaussianBlur(input_image, kernel, sigmaX=0)
    elif blurring == 'median':
        return cv2.medianBlur(input_image, kernel_size)
    elif blurring == 'bilateral':
        return cv2.bilateralFilter(input_image, d=kernel_size, sigmaColor=75, sigmaSpace=75)
    else:
        return np.zeros(shape=input_image.shape)

# Removes noise from the image.
# Params: filter_strength color_strength
def denoise(input_image, params):
    filter_strength, color_strength = int( params[0] ), int( params[1] )
    return cv2.fastNlMeansDenoisingColored(input_image, h=filter_strength, hColor=color_strength,
                                            templateWindowSize=21, searchWindowSize=7)

# Sharpens with edges of the image.
# Params: None
def sharpen(input_image, params):
    sharpening_kernel = np.array( [
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ] )
    return cv2.filter2D(input_image, ddepth=-1, kernel=sharpening_kernel)

# Sharpens with edges of the image.
# Params: mean/otsu/gaussian
def threshold(input_image, params):
    thresholding = params[0]
    gray_image = cv2.cvtColor(src=input_image, code=cv2.COLOR_BGR2GRAY)
    if thresholding == 'mean':
        return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=3, C=5)
    elif thresholding == 'otsu':
        return cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresholding == 'gaussian':
        return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=3, C=5)
    else:
        return np.zeros(shape=input_image.shape)

# Dilation (Enhancement) of the edges in the image.
# Params: kernel_size
def dilate(input_image, params):
    kernel_size = int( params[0] )
    kernel = (kernel_size, kernel_size)
    return cv2.dilate(input_image, kernel, iterations=1)


# Dilation (Disenhancement) of the edges in the image.
# Params: kernel_size
def erode(input_image, params):
    kernel_size = int( params[0] )
    kernel = (kernel_size, kernel_size)
    return cv2.erode(input_image, kernel, iterations=1)

# Applys morphological transformation on the image for removing noise.
# Params: opening/closing kernel_size
def morphology(input_image, params):
    morph = params[0]
    kernel_size = int( params[1] )
    kernel = (kernel_size, kernel_size)
    if morph == 'opening':
        return cv2.morphologyEx(input_image, cv2.MORPH_OPEN, kernel)
    elif morph == 'closing':
        return cv2.morphologyEx(input_image, cv2.MORPH_CLOSE, kernel)
    else:
        return np.zeros(shape=input_image.shape)

# Uses canny's edge-detection algorithm to detect edges in the image.
# Params: threshold1 threshold2
def edge_detect(input_image, params):
    h1, h2 = int( params[0] ), int( params[1] )
    return cv2.Canny(input_image, threshold1=h1, threshold2=h2)

# Applys a perspective transformation on a region of the image.
# Params x1 y1 x2 y2 x3 y3 x4 y4 x1' y1' x2' y2' x3' y3' x4' y4'.
def perspective(input_image, params):
    height, width = input_image.shape[:2]
    src_x1, src_y1, src_x2, src_y2, src_x3, src_y3, src_x4, src_y4 = [ int(param) for param in params[0:8] ]
    dest_x1, dest_y1, dest_x2, dest_y2, dest_x3, dest_y3, dest_x4, dest_y4 = [ int(param) for param in params[8:] ]
    src_points = np.float32( [
        [src_x1, src_y1],
        [src_x2, src_y2],
        [src_x3, src_y3],
        [src_x4, src_y4]
    ] )
    dest_points = np.float32( [
        [dest_x1, dest_y1],
        [dest_x2, dest_y2],
        [dest_x3, dest_y3],
        [dest_x4, dest_y4]
    ] )
    M = cv2.getPerspectiveTransform(src_points, dest_points)
    return cv2.warpPerspective( input_image, M, (height, width) )

algorithm_dict = {
    'rgb2gray' : 0,
    'hist' : 1,
    'draw' : 2,
    'translate' : 3,
    'rotate' : 4,
    'resize': 5,
    'crop' : 6,
    'brighten' : 7,
    'darken' : 8,
    'blur' : 9,
    'denoise' : 10,
    'sharpen' : 11,
    'threshold' : 12,
    'dilate' : 13,
    'erode' : 14,
    'morphology' : 15,
    'edge_detect' : 16,
    'perspective' : 17
}

algorithm_list = [
    rgb2gray,
    hist,
    draw,
    translate,
    rotate,
    resize,
    crop,
    brighten,
    darken,
    blur,
    denoise,
    sharpen,
    threshold,
    dilate,
    erode,
    morphology,
    edge_detect,
    perspective
]

def main(argv):
    algorithm_name = argv[0]
    input_filepath = argv[1]
    output_filepath = argv[2]
    
    params = argv[3:] if len(argv) > 2 else []

    # Loading input image.
    input_image = cv2.imread(input_filepath)

    # Applying algorithm to input image.
    output_image = algorithm_list[ algorithm_dict[algorithm_name] ](input_image, params)

    # Showing image.
    cv2.imshow(output_filepath, output_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Storing output image.
    cv2.imwrite(output_filepath, output_image)

if __name__ == "__main__":
    main(sys.argv[1:])