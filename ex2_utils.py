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
    k_height, k_width = kernel.shape
    im_height, im_width = in_image.shape
    # Pads with the edge values of array.
    image_padded = np.pad(in_image, ((k_height // 2, k_width // 2),(k_height // 2, k_width // 2)), 'edge')
    # create new matrix that will be our answer
    convolved_mat = np.zeros((im_height, im_width))
    # Multiply each cell in the kernel with its parallel cell in the image matrix
    # And sum all the multiplicities and place the sum in the output matrix at (x,y)
    for i in range(im_height):
        for j in range(im_width):
            convolved_mat[i, j] = np.round((image_padded[i:i + k_height, j:j + k_width] * kernel).sum())
    return convolved_mat




def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale image
    :return: (directions, magnitude)
    """
# # As we learned, an image gradient is a directional change in the intensity or color in an image
#
# # First, we will define the kernel as required
#     kernel = np.array([[1, 0, -1]])
# # To compute the directions, we will
#     xDirection= conv2D(in_image, kernel)
#     yDirection= conv2D(in_image,kernel.T) # send the transpose of the kernel array
#     directions = np.arctan2(yDirection, xDirection)

    #############################
    # kernel1 = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    # kernel2 = kernel1.transpose()
    # x_der = conv2D(inImage, kernel1)
    # y_der = conv2D(inImage, kernel2)
    # directrions = np.arctan(y_der, x_der)
    # mangitude = np.sqrt(np.square(x_der) + np.square(y_der))
    ################################
# Now , to compute the magnitude, we will use the formula as we learned in class:
#     #Magnitude=square_root((Change in the X-axis)^2+(Change in the Y-axis)^2)
#     magnitude=math.sqrt(xDirection**2+ yDirection**2)

    # apply conv2D on in_image with kernel [-1,0,1]
    x_der=cv2.filter2D(in_image,-1,np.array([-1,0,1]), borderType=cv2.BORDER_REPLICATE)
    # apply conv2D on in_image with kernel [[-1],[0],[1]]
    y_der=cv2.filter2D(in_image,-1,np.array([[-1],[0],[1]]), borderType=cv2.BORDER_REPLICATE)
    magnitude = np.sqrt(np.square(x_der) + np.square(y_der))
    directions = np.arctan(y_der/ x_der)

    return directions, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    # gaussian = gaussKernel(k_size) #send the kernel size to get a row of the binomial coefficients.
    # ans= conv2D(in_image, gaussian) #apply 2D convolution on the Input image with the gaussian kernel we found

    # sigma = int(round(0.3 * ((k_size - 1) * 0.5 - 1) + 0.8))
    # kernel = createGaussianKer(k_size, sigma)
    # ans= conv2D(in_image, kernel)
    # return ans

    # sig = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    # ax = np.linspace(-(k_size - 1) / 2., (k_size - 1) / 2., k_size)
    # xx, yy = np.meshgrid(ax, ax)
    # kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    # img = conv2D(in_image, kernel)
    # return img

    sigma = 1#int(round(0.3 * ((k_size - 1) * 0.5 - 1) + 0.8))
    #gauss_kernel = createGaussianKer(k_size)
    gauss_kernel= gaussKernel(k_size)
    # apply conv2D on in_image with gauss_kernel
    ans=cv2.filter2D(in_image,-1,gauss_kernel, borderType=cv2.BORDER_REPLICATE)
    return ans


def createGaussianKer(kernel_size):
    center=(int)(kernel_size/2)
    kernel=np.zeros((kernel_size,kernel_size))
    for i in range(kernel_size):
       for j in range(kernel_size):
          diff=np.sqrt((i-center)**2+(j-center)**2)
          kernel[i,j]=np.exp(-(diff**2)/(2))
          kernel[i,j]=np.exp(-(diff**2)/(2*math.pi)) #defoltive sigma equal to 1 , so there is no need to put it here

    return kernel/np.sum(kernel)

def gaussKernel(k_size: int) -> np.ndarray:
    """
    :param k_size: Kernel size
    :return: A row of the binomial coefficients.
    """
    #As it written in the PDF ,a consequent 1D convolution of [1 1] with itself is an elegant way
    # for obtaining a row of the binomial coefficients.Therefore:
    g = np.array([1, 1]) #define the array
    gaussian = np.array(g) #again with different variable
    for i in range(k_size - 2):
        gaussian = conv1D(g, gaussian) #send [1 1] with itself to 1D convolution
    gaussian = np.array([gaussian])
    #multiply gaussian with the transpose of gaussian
    gaussian = gaussian.T.dot(gaussian)
    #divide each element by the sum of elements in gaussian
    return gaussian / gaussian.sum()

def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    # I used the website: https://quick-adviser.com/what-is-sigma-in-gaussian-filter/
    # How do you set the Sigma in Gaussian filter?
    # The rule of thumb for Gaussian filter design is to choose the filter size to be about 3 times the standard deviation
    # (sigma value) in each direction, for a total filter size of approximately 6*sigma rounded to an odd integer value.
    #Therefore:
    sigma = 1#0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    #by using pythons internal function 'getGaussianKernel', we will creates Gaussian kernel
    gaussian_kernel=cv2.getGaussianKernel(k_size,sigma)
    # by using pythons internal function 'filter2D', we will apply the gaussian_kernel on the image
    return cv2.filter2D(in_image, -1, gaussian_kernel, borderType=cv2.BORDER_REPLICATE)

def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    #I did not chose to implement this function, but i want to wrote it in case i will need it ,
    # therefore I put it in comment
    # by using: http://portal.survey.ntua.gr/main/labs/rsens/DeCETI/IRIT/GIS-IMPROVING/node18.html
    # we will define the laplacian kernel by using pythons internal function:
    laplacian_kernel = cv2.Laplacian()# this is the following array: ([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    conv_img = conv2D(img, laplacian_kernel) #Apply the laplacian_kernel on the input image
    # find the edges in conv_img
    edgeMat= find_crossing_zero(conv_img)
    return edgeMat



def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    #by using: https://www.youtube.com/watch?v=uNP6ZwQ3r6A&t=202s
   #we will define the laplacian kernel by using pythons internal function:
    laplacian_kernel=cv2.Laplacian()  # this is the following array: ([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    # we will find the gauss kernel by using k_size we found by trying various values
    gauss_Kernel = gaussKernel(15)
    gaussian_laplacian = conv2D(gauss_Kernel, laplacian_kernel)
    conv_img = conv2D(img, gaussian_laplacian)#Apply the laplacian gaussian kernel on the input image
    # find the edges in conv_img
    edgeMat=find_crossing_zero(conv_img)
    return edgeMat

def find_crossing_zero(img: np.ndarray) -> np.ndarray:
    """
        find edges by find the + 0 - or + -
    :param img: given image
    :return: edge matrix
    """
    # We will take the size of the image
    height, width=img.shape[:2]
    # And initial matrix of zeros in the same size
    edgeMat= np.zeros(img.shape)
    # We will move over all pixels in image
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # if there is zero crossing:
            if (((img[i - 1, j] > 0 and img[i + 1, j] < 0) or (img[i - 1, j] < 0 and img[i + 1, j] > 0)) or
                    ((img[i, j - 1] > 0 and img[i, j + 1] < 0) or (img[i, j - 1] < 0 and img[i, j + 1] > 0))):
                # we will define this cell in edge matrix as 1
                edgeMat[i,j]=1

            elif img[i, j] < 0:
                if img[i - 1, j] > 0:
                    edgeMat[i - 1, j] = 1
                elif img[i + 1, j] > 0:
                    edgeMat[i + 1, j] = 1
                elif img[i, j - 1] > 0:
                    edgeMat[i, j - 1] = 1
                elif img[i, j + 1] > 0:
                    edgeMat[i, j + 1] = 1

            elif img[i, j] > 0:
                if ((img[i - 1, j] < 0 or img[i + 1, j] < 0)
                        or (img[i, j - 1] < 0 or img[i, j + 1] < 0)):
                    edgeMat[i, j] = 1
# return the matrix of edges where the pixel that represent zero crossing will be 1 and pixel that does not will be 0
    return edgeMat

    # def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    # """
    # Find Circles in an image using a Hough Transform algorithm extension
    # To find Edges you can Use OpenCV function: cv2.Canny
    # :param img: Input image
    # :param min_radius: Minimum circle radius
    # :param max_radius: Maximum circle radius
    # :return: A list containing the detected circles,
    #             [(x,y,radius),(x,y,radius),...]
    # """
    #
    # return


# def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
#         np.ndarray, np.ndarray):
#     """
#     :param in_image: input image
#     :param k_size: Kernel size
#     :param sigma_color: represents the filter sigma in the color space.
#     :param sigma_space: represents the filter sigma in the coordinate.
#     :return: OpenCV implementation, my implementation
#     """
#
#     return