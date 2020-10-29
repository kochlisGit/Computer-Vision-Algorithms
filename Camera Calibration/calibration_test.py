import cv2
import pickle

def load_settings(filepath):
    pickle_in = open(filepath, 'rb')
    data = pickle.load(pickle_in)
    mtx = data['mtx']
    dist = data['dist']
    return mtx, dist

def apply_undistortion(filepath, mtx, dist):
    original_image = cv2.imread(filepath)
    undistored_image = cv2.undistort(original_image, mtx, dist, None, mtx)

    original_image = cv2.resize( original_image, dsize=(original_image.shape[1] // 2, original_image.shape[0] // 2) )
    undistored_image = cv2.resize( undistored_image, dsize=(undistored_image.shape[1] // 2, undistored_image.shape[0] // 2) )

    cv2.imshow('original image', original_image)
    cv2.imshow('undistored image', undistored_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

mtx, dist = load_settings('settings.pickle')
apply_undistortion('images/test_image.JPEG', mtx, dist)