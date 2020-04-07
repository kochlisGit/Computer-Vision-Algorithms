# Sharpening

I have implemented 2 methods for sharpening an image. You can notice how sharpening is applied in an image by looking at the edges. Basically, sharpening is the enchancement of the edges.

# Laplacian Filter

![Image of Laplacian Filter](https://wikimedia.org/api/rest_v1/media/math/render/svg/beb8b9a493e8b9cf5deccd61bd845a59ea2e62cc)

By apply the laplacian kernel we can achieve image sharpening. I used the opencv library to apply the kernel to the image with the fastest way possible:

    return cv.filter2D(image, -1, sharpen_kernel)
    
# Unsharp Mark
In this method, we create the **Gaussian Blur Mark** and subtract it from the original image. This will remove noise, because Laplacian Filter is sensitive to noise. Then we apply the Laplacian Kernel. This method is also called **LoG** or **Laplacian of Gaussians**.

You can find more information of how this method works here: https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
