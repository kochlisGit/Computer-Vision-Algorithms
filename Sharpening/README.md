# Sharpening

I have implemented 2 methods for sharpening an image. You can notice how sharpening is applied in an image by looking at the edges. Basically, sharpening is the enchancement of the edges.

# Laplacian Filter

![Image of Laplacian Filter](https://wikimedia.org/api/rest_v1/media/math/render/svg/beb8b9a493e8b9cf5deccd61bd845a59ea2e62cc)

By apply the laplacian kernel we can achieve image sharpening. I used the opencv library to apply the kernel to the image with the fastest way possible:

    return cv.filter2D(image, -1, sharpen_kernel)
    
# Unsharp Mark
In this method, we create the **Gaussian Blur Mark** and subtract it from the original image. This will remove noise, because Laplacian Filter is sensitive to noise. Then we apply the Laplacian Kernel. This method is also called **LoG** or **Laplacian of Gaussians**. Also, You can control the amount of the sharpening applied in the image:

    sharpen_image_with_unsharp_mask(image, kernel_size = (5, 5), sigma = 1.0, amount = 1.0, threshold = 0)
    
You can also tune the parameters to see the results in the image.

You can find more information of how this method works here: https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm

# Results
Below You can compare the original image with the sharpened image using these 2 different methods.

Original Image

![Original Image](https://github.com/kochlisGit/Computer-Vision-Algorithms/blob/master/Sharpening/Database/Cat.jpg)

Sharpened with Laplacian Kernel

![Sharpened Image](https://github.com/kochlisGit/Computer-Vision-Algorithms/blob/master/Sharpening/cat_sharpened_filter.jpg)

Sharpened with Unsharp

![Sharpened Image](https://github.com/kochlisGit/Computer-Vision-Algorithms/blob/master/Sharpening/cat_sharpened_unmask.jpg)

There are also more pictures in the Database folder, which You an try to enchance Yourself!
