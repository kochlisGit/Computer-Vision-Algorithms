# Image Distortion
It is known that there is no perfect camera. Every camera adds some distortion to the original image. A good camera, will add little distortion to the image.
The distortion affects the curvature of the lines in the image. The difference between the original image and the distorted image is so small that You can barely notice it.
However, there are some computer vision examples, where You would like to know the true curvature of the lines.

For example, a self-driving car should perceive the world with the same way as a driver would do. A camera without calibration, would probably
make the curvature of the lane look a little different than it actually is. As a result, the car would probably take wrong actions, because It would perceive the road lane
differently.

# Calibrating the Image
The easiest way to calibrate a camera is to capture Chessboard photos, then use opencv to find the chessboard's corners and calibrate the camera. In this example, 
I used my phone's camera to take pictures of a chessboard. It is important to take the pictures from a different perspective each time.

![Chessboard image1](https://github.com/kochlisGit/Computer-Vision-Algorithms/blob/master/Camera%20Calibration/images/chessboard1.JPEG)
![Chessboard image2](https://github.com/kochlisGit/Computer-Vision-Algorithms/blob/master/Camera%20Calibration/images/chessboard2.JPEG)
![Chessboard image3](https://github.com/kochlisGit/Computer-Vision-Algorithms/blob/master/Camera%20Calibration/images/chessboard3.JPEG)
![Chessboard image4](https://github.com/kochlisGit/Computer-Vision-Algorithms/blob/master/Camera%20Calibration/images/chessboard4.JPEG)
![Chessboard image5](https://github.com/kochlisGit/Computer-Vision-Algorithms/blob/master/Camera%20Calibration/images/chessboard5.JPEG)

# Calibration with OpenCV
OpenCV offers 3 important functions for image undistortion:

    1. cv2.findChessboardCorners(image, patternSize, corners=None, flags=None)
    2. cv2.calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix=None, distCoeffs=None)
    3. cv2.undistort(src, cameraMatrix, distCoeffs, dst, newCameraMatrix)
    
# Results
You can see below the comparison between the distored image (on the left) and the undistorted image (on the right).

![comparison image](https://github.com/kochlisGit/Computer-Vision-Algorithms/blob/master/Camera%20Calibration/images/comparison.png)

In the first Image, which is distorted, It looks as If the car is perfectly positioned on the lane and should move forward. However, In the second image, which shows
how the vehicle is really positioned on the lane, You can clearly notice that the car should correct It's steering (perhaps 5 angles to the right).
