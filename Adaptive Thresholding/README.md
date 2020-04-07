# Adaptive Thresholding
I have Implemented the wellner's algorithm for thresholding a grayscale image, with 2 important optimizations. The main idea
in Wellner’s algorithm is that each pixel is compared to an average of the surrounding pixels. Specifically, an
approximate moving average of the last s pixels seen is calculated while traversing the image. If the value of the
current pixel is t percent lower than the average then it is set to black, otherwise it is set to white. If You set s = 1/8 of image's width and t = 15, then this algorithm can produce some very good results. However, a problem with this method is that it is
dependent on the scanning order of the pixels. In addition, the moving average is not a good representation of the
surrounding pixels at each step because the neighbourhood samples are not evenly distributed in all directions.
So, the first optimization is that I sacrificed 1 iteration to an integral image. Then. by using the integral image, I compute the average of an s x s window of pixels centered around each pixel. This is a better average for comparison, since it considers neighbouring pixels on all sides and It can be done in Linear time using the integral image.

# Performance & Results
This method is 2 times slower than Wellner's algorithm. However, It will produce better results in an Image with strong Illumination changes. With some Normal hardware, You can achieve real-time thresholding.
