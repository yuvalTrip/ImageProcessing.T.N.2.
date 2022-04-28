import math
import numpy as np
import cv2

def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 318916335
#Helpful links I used:
# https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
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
        #print(row)
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
# As we learned, an image gradient is a directional change in the intensity or color in an image
    #We will strat by compute the directions:
    # apply conv2D on in_image with kernel [-1,0,1]
    x_der=cv2.filter2D(in_image,-1,np.array([-1,0,1]), borderType=cv2.BORDER_REPLICATE)
    # apply conv2D on in_image with kernel [[-1],[0],[1]]
    y_der=cv2.filter2D(in_image,-1,np.array([[-1],[0],[1]]), borderType=cv2.BORDER_REPLICATE)
    #Compute the Magnitude:=square_root((Change in the X-axis)^2+(Change in the Y-axis)^2)
    magnitude = np.sqrt(np.square(x_der) + np.square(y_der))
    directions = np.arctan2(y_der, x_der)

    return directions, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    gauss_kernel= gaussKernel(k_size)#send the kernel size to get a row of the binomial coefficients.
    # apply conv2D on in_image with gauss_kernel
    ans=cv2.filter2D(in_image,-1,gauss_kernel, borderType=cv2.BORDER_REPLICATE)#apply 2D convolution on the Input image with the gaussian kernel we found
    return ans

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
    #Therefore it sopposed to be: sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    #BUT in tirgul we learned that we can choose sigma as 1, such as defoltieve value
    sigma = 1
    #by using pythons internal function 'getGaussianKernel', we will creates Gaussian kernel
    gaussian_kernel=cv2.getGaussianKernel(k_size,sigma)
    # by using pythons internal function 'filter2D', we will apply the gaussian_kernel on the image (like conv2D function)
    return cv2.filter2D(in_image, -1, gaussian_kernel, borderType=cv2.BORDER_REPLICATE)

def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    pass
    # #I did not chose to implement this function, but i want to wrote it in case i will need it ,
    # # therefore I put it in comment
    # # by using: http://portal.survey.ntua.gr/main/labs/rsens/DeCETI/IRIT/GIS-IMPROVING/node18.html
    # # we will define the laplacian kernel by using pythons internal function:
    # laplacian_kernel =  np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    # # by using pythons internal function 'filter2D', we will apply the laplacian_kernel on the input image (like conv2D function)
    # conv_img = cv2.filter2D(img, -1, laplacian_kernel, borderType=cv2.BORDER_REPLICATE)
    #
    # #conv_img = conv2D(img, laplacian_kernel) #Apply the laplacian_kernel on the input image
    # # find the edges in conv_img
    # edgeMat= find_crossing_zero(conv_img)
    # return edgeMat



def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    #by using: https://www.youtube.com/watch?v=uNP6ZwQ3r6A&t=202s
   #we will define the laplacian kernel by using pythons internal function:

    laplacian_kernel=np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])#cv2.Laplacian()  # this is the following array: ([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    # we will find the gauss kernel by using k_size we found by trying various values
    gauss_Kernel = gaussKernel(15)

    # by using pythons internal function 'filter2D', we will apply the laplacian_kernel on the gauss_Kernel (like conv2D function)
    gaussian_laplacian=cv2.filter2D(gauss_Kernel, -1, laplacian_kernel, borderType=cv2.BORDER_REPLICATE)
    #gaussian_laplacian = conv2D(gauss_Kernel, laplacian_kernel)

    # by using pythons internal function 'filter2D', we will apply the laplacian_kernel on the input image (like conv2D function)
    conv_img = cv2.filter2D(img, -1, gaussian_laplacian, borderType=cv2.BORDER_REPLICATE)

    #conv_img = conv2D(img, gaussian_laplacian)#Apply the laplacian gaussian kernel on the input image
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
#More conditions to find zero crossing:
            elif img[i, j] < 0:
                if img[i - 1, j] > 0:
                    edgeMat[i - 1, j] = 1
                elif img[i, j - 1] > 0:
                    edgeMat[i, j - 1] = 1
                elif img[i + 1, j] > 0:
                    edgeMat[i + 1, j] = 1
                elif img[i, j + 1] > 0:
                    edgeMat[i, j + 1] = 1
# One last condition to find zero crossing:
            elif img[i, j] > 0:
                if ((img[i - 1, j] < 0 or img[i + 1, j] < 0) or (img[i, j - 1] < 0 or img[i, j + 1] < 0)):
                    edgeMat[i, j] = 1
# return the matrix of edges where the pixel that represent zero crossing will be 1 and pixel that does not will be 0
    return edgeMat

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
    # Using OpenCV Canny Edge detector to detect edges
    # The arguments are: (Source/Input image, the High threshold value of intensity gradient,the Low threshold value of intensity gradient)

    edged_image = cv2.Canny(np.uint8(img * 255), 200, 400)
    edged_image = edged_image / 255 #normlize the values to be 0 or 1

    #display the image
    #cv2.imshow("", edged_image)
    #cv2.waitKey(0)

    #Initial an 3D matrix of zeros
    circle_counter = np.zeros((img.shape[0], img.shape[1], max_radius - min_radius + 1))

    # We will move over all pixels in image
    for x in range(edged_image.shape[0]):
        for y in range(edged_image.shape[1]):
            #If there is edge in specific cell
                if edged_image[x][y] == 1:
            #we will check for each angle
                 for teta in range(360):
                    y_teta = np.sin(np.deg2rad(teta)) #polar coordinate for center (convert to radians)- x values
                    x_teta = np.cos(np.deg2rad(teta)) #polar coordinate for center (convert to radians)- y values
                # now we will calculate the parameters of the circle
                    #we will check for each radius in the given range
                    for radius in range(min_radius, max_radius):
                        yCenter = y - (radius * y_teta)
                        xCenter = x - (radius * x_teta)
                        if xCenter > 0 and yCenter > 0:
                            try:
                                circle_counter[int(np.round(xCenter))][int(np.round(yCenter))][radius - min_radius] += 1  #voting for this circle in circle_counter matrix
                            except Exception:#if one of the values is negative, we will ignore it
                                continue
# Now we will select circles by treshold,
# and create list of circles we will return at the end , every parameter will be (x_center,Ycenter,radius)
    mostVoted = np.argwhere(circle_counter >= (np.max(circle_counter) - np.max(circle_counter) * 0.32))  # take the 30% most voted
    mostVoted = mostVoted[:, [1, 0, 2]]
    mostVoted[:, 2] += min_radius  # put the original radiuses of the circles

    return list(map(tuple, mostVoted))



def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    #I inspired by using this github:
    #https://github.com/anlcnydn/bilateral/blob/25b238f42cee6dad13a3e834d5fdd3860a456aa3/bilateral_filter.py#L40
    open_cv_implement=cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    #The 'image' 'np.array' must be given gray-scale. It is suggested that to use OpenCV.??????????????????
    #my_implement
    #First we will create new matrix for the bilateral filtered image
    bilateralIm=np.zeros(in_image.shape)

    padIm=np.pad(in_image,((k_size//2,k_size//2), (k_size//2,k_size//2)),mode="edge")
    #Iterate over all pixels in image
    for x in range ((k_size//2), padIm.shape[0]- k_size//2):
        for y in range ((k_size//2),padIm.shape[1]-k_size//2):
            sumOfNumerator=0
            sumOfDenominator=0
            for kernelX in range (-(k_size//2), (k_size//2)+1):
                for kernelY in range (-(k_size//2), (k_size//2)+1):
                    gauss_c=gaussian(distance((x,y),(x+kernelX,y+kernelY)),sigma_color)
                    gauss_s=gaussian(abs(padIm[x+kernelX][y+kernelY]-padIm[x][y]),sigma_space)
                    #Sum the Numerator and the Denominator as we learned in class (by the formula in Lecture 2, Slide 130)
                    sumOfNumerator=sumOfNumerator+(gauss_s*gauss_c*padIm[x+kernelX][y+kernelY])
                    sumOfDenominator=sumOfDenominator+(gauss_s*gauss_c)

            bilateralIm[x-(k_size//2),y-(k_size//2)]=np.round(sumOfNumerator/sumOfDenominator).astype(int)
    my_implement=bilateralIm

    return open_cv_implement,my_implement

def gaussian(x, sigma):
    """
    Function compute Ws(p) and Wc(p) according to the formuls we learned in Lecture 2, Slide 130
    :param x:given vector
    :param sigma: given sigma
    :return:I'(x) in single iteration
    """
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))

def distance(point1, point2):
    """
    Function compute distance between 2 points.
    :param point1: (x1,y1)
    :param point2: (x2,y2)
    :return: distance
    """
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
