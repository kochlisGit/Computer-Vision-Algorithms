# Image Distortion
It is known that there is no perfect camera. Every camera adds some distortion to the original image. A good camera, will add little distortion to the image.
The distortion affects the curvature of the lines in the image. The difference between the original image and the distorted image is so small that You can barely notice it.
However, there are some computer vision examples, where You would like to know the true curvature of the lines.

For example, a self-driving car should perceive the world with the same way as a driver would do. A camera without calibration, would probably
make the curvature of the lane look a little different than it actually is. As a result, the car would probably take wrong actions, because It would perceive the road lane
differently.

# Calibrating the Image
The easiest way to calibrate a camera is to capture Chessboard photos, then use opencv to find the chessboard's corners and calibrate the camera. In this example, 
I used my phone's camera to take pictures of a chessboard.

