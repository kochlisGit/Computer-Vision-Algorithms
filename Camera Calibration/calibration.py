import cv2
import numpy as np
import glob
import pickle

# Termination criteria for the calibration.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Camera calibration using chessboard images.
def calibration(image_dir, image_name, height=7, width=7):
    images = glob.glob(image_dir + image_name + '*.JPEG')

    world_points = []
    image_points = []
    corner_points = np.zeros( (height*width, 3), np.float32 )
    corner_points[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    for image_path in images:
        # Loading an image.
        input_image = cv2.imread(image_path)

        # Converting image to grayscale.
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

        # Find the chess board corners
        pattern_found, corners = cv2.findChessboardCorners(gray_image, (width, height), None)

        if pattern_found:
            world_points.append(corner_points)
            image_points.append(corners)

    pattern, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        world_points,
        image_points,
        gray_image.shape[::-1],
        None,
        None
    )
    return (pattern, mtx, dist, rvecs, tvecs)

# Calibrating the image.
pattern, mtx, dist, rvecs, tvecs = calibration('images/', 'chessboard', height=7, width=7)

# Saving the calibration settings.
calibration_dict = {'mtx': mtx, 'dist': dist}

pickle_out = open('settings.pickle', 'wb')
pickle.dump(calibration_dict, pickle_out)
pickle_out.close()