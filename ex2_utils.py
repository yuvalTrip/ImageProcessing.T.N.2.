import math
import numpy as np
import cv2


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
#As we learned in tirgul, the steps for convolution are:
    #take a vector (the kernel, k_size in this case)
    #Flip it
    #Move it over the other vector
    #Multiply the values
# First we will define new array of zeroes in size of both given arrays
    ans = np.zeros(len(in_signal) + len(k_size) - 1)
#First we will flip the kernel:
    k_size = np.flip(k_size) #for example : [1,2,3]->[3,2,1]

    # now we will pad the  with 0 - for example if our in_signal is [1,1] we will make it:
    #[0,0,1,1,0,0]. We will pad each side with len(k_size) - 1 zeroes
    #because we want the first element of kernel to be with the first element of in_signal
    pad_in_signal = np.pad(in_signal, (len(k_size) - 1, len(k_size) - 1), 'constant')

# Now we will move over both arrays and multiply ans sum the relevant values.
    j = 0
    for i in range(len(pad_in_signal) - (len(k_size) - 1)):
        ans[j] = ((k_size * pad_in_signal[i:i + len(k_size)]).sum())
        j = j+ 1

    return ans
def flipped_mat(kernel: np.ndarray) -> np.ndarray:
    """

    :param kernel: A kernel
    :return: A flipped matrix
    """
# we will flip each row , then do transpose on the matrix and than flip again all rows and than transpose again
    i = 0
    for row in kernel:
        row = np.flip(row)  # for example : [1,2,3]->[3,2,1]
        print(row)
        kernel[i] = row
        i = i + 1
    # now transpose
    kernel_transpose = kernel.transpose()
    # now flip again each row (because now those are columns)
    i = 0
    for row in kernel_transpose:
        row = np.flip(row)  # for example : [1,2,3]->[3,2,1]
        kernel_transpose[i] = row
        i = i + 1

    kernel_transpose = kernel_transpose.transpose()
    return kernel_transpose

def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    kernel_flipped=flipped_mat(kernel)
    # now we have flipped matrix
    #we will take the shape of the kernel and the in_image
    k_height, k_width = kernel_flipped.shape
    im_height, im_width = in_image.shape
    # Pads with the edge values of array.
    padded_mat = np.pad(in_image, ((k_height, k_height), (k_width, k_width)), 'edge')

#create new matrix that will be our answer
    convolved_mat = np.zeros((im_height, im_width), dtype="float32")
#Multiply each cell in the kernel with its parallel cell in the image matrix
#And sum all the multiplicities and place the sum in the output matrix at (x,y)

    for i in range(im_height):
        for j in range(im_width):
            x_head = j + 1 + k_width
            y_head = i + 1 + k_height
            convolved_mat[i, j] = (padded_mat[y_head:y_head + k_height, x_head:x_head + k_width] * kernel_flipped).sum()

    return convolved_mat

def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale image
    :return: (directions, magnitude)
    """
# As we learned, an image gradient is a directional change in the intensity or color in an image

# First, we will define the kernel as required
    kernel = np.array([[1, 0, -1]])
# To compute the directions, we will
    xDirection= conv2D(in_image, kernel)
    yDirection= conv2D(in_image,kernel.T)
    directions = np.arctan2(yDirection, xDirection)

    #############################
    # kernel1 = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    # kernel2 = kernel1.transpose()
    # x_der = conv2D(inImage, kernel1)
    # y_der = conv2D(inImage, kernel2)
    # directrions = np.arctan(y_der, x_der)
    # mangitude = np.sqrt(np.square(x_der) + np.square(y_der))
    ################################
# Now , to compute the magnitude, we will use the formula as we learned in class:
    #Magnitude=square_root((Change in the X-axis)^2+(Change in the Y-axis)^2)
    magnitude=math.sqrt(xDirection**2+ yDirection**2)

    return directions, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
# I used the website: https://quick-adviser.com/what-is-sigma-in-gaussian-filter/
# How do you set the Sigma in Gaussian filter?
# The rule of thumb for Gaussian filter design is to choose the filter size to be about 3 times the standard deviation
# (sigma value) in each direction, for a total filter size of approximately 6*sigma rounded to an odd integer value.

# Basically, what we actually need to do is to use the convolution function, by sending the in_image we have , and
    (in_image: np.ndarray, kernel: np.ndarray)

    return


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return