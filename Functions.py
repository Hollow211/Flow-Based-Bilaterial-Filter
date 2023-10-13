# pip install strcture-tensor
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from numpy.linalg import norm
from math import floor, pi, exp, ceil
from numpy.linalg import eig
import scipy
from structure_tensor import eig_special_2d, structure_tensor_2d


def Mystructure_tensor_2d(image, sigma, rho, out=None, truncate=4.0):
    # Make sure it's a Numpy array.
    image = np.asarray(image)

    # Check data type. Must be floating point.
    if not np.issubdtype(image.dtype, np.floating):
        logging.warning(
            'image is not floating type array. This may result in a loss of precision and unexpected behavior.')

    IxL = ndimage.gaussian_filter(image[:, :, 0], sigma, order=[1, 0], mode='nearest', truncate=truncate)
    IxA = ndimage.gaussian_filter(image[:, :, 1], sigma, order=[1, 0], mode='nearest', truncate=truncate)
    IxB = ndimage.gaussian_filter(image[:, :, 2], sigma, order=[1, 0], mode='nearest', truncate=truncate)

    IyL = ndimage.gaussian_filter(image[:, :, 0], sigma, order=[0, 1], mode='nearest', truncate=truncate)
    IyA = ndimage.gaussian_filter(image[:, :, 1], sigma, order=[0, 1], mode='nearest', truncate=truncate)
    IyB = ndimage.gaussian_filter(image[:, :, 2], sigma, order=[0, 1], mode='nearest', truncate=truncate)

    # Compute derivatives (Scipy implementation truncates filter at 4 sigma).
    Ix = IxL + IxA + IxB
    Iy = IyL + IyA + IyB
    if out is None:
        # Allocate S.
        S = np.empty((3, image.shape[0], image.shape[1]), dtype=image.dtype)
    else:
        # S is already allocated. We assume the size is correct.
        S = out

    # Integrate elements of structure tensor (Scipy uses sequence of 1D).
    tmp = np.empty((image.shape[0], image.shape[1]), dtype=image.dtype)
    np.multiply(Ix, Ix, out=tmp)
    ndimage.gaussian_filter(tmp, rho, mode='nearest', output=S[0], truncate=truncate)
    np.multiply(Iy, Iy, out=tmp)
    ndimage.gaussian_filter(tmp, rho, mode='nearest', output=S[1], truncate=truncate)
    np.multiply(Ix, Iy, out=tmp)
    ndimage.gaussian_filter(tmp, rho, mode='nearest', output=S[2], truncate=truncate)

    return S


def BilaterialPass(img, v2, RenderPass, sigmaD, sigmaR):
    height, width = img.shape[0], img.shape[1]

    # Create arrays to represent the x and y coordinates
    x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    ## START a LOOP ###
    uv = np.stack((x, y), axis=-1)  # s.p

    dir = v2.copy()

    if RenderPass == 0:
        tmp = dir[:, :, 0].copy()
        dir[:, :, 0] = dir[:, :, 1]
        dir[:, :, 1] = -1 * tmp

    dabs = np.abs(dir)
    ds = 1.0 / np.maximum(dabs[:, :, 0], dabs[:, :, 1])

    center = img.copy()
    sum = img.copy()
    norm = np.ones((height, width), dtype='float32')

    kernelSize = 2.0 * sigmaD

    for i in np.arange(1, kernelSize + 1):
        dir_step_i = ds * i

        index0 = (uv + dir_step_i[:, :, np.newaxis] * dir).astype('int')

        px = np.clip(index0[:, :, 0], 0, height - 1)
        py = np.clip(index0[:, :, 1], 0, width - 1)

        c0 = img[px, py]
        ############
        index1 = (uv - dir_step_i[:, :, np.newaxis] * dir).astype('int')

        px1 = np.clip(index1[:, :, 0], 0, height - 1)
        py1 = np.clip(index1[:, :, 1], 0, width - 1)

        c1 = img[px1, py1]

        e0 = np.linalg.norm(c0 - center, axis=2)
        e1 = np.linalg.norm(c1 - center, axis=2)

        kerneld = np.exp(-dir_step_i * dir_step_i / (2.0 * sigmaD ** 2))
        kernele0 = np.exp(-e0 ** 2 / (2.0 * sigmaR ** 2))
        kernele1 = np.exp(-e1 ** 2 / (2.0 * sigmaR ** 2))

        norm += kerneld * kernele0
        norm += kerneld * kernele1

        sum += kerneld[:, :, np.newaxis] * kernele0[:, :, np.newaxis] * c0
        sum += kerneld[:, :, np.newaxis] * kernele1[:, :, np.newaxis] * c1

    sum /= norm[:, :, np.newaxis]

    return sum


def FlowBilaterial(img, v2, sigmaD, sigmaR):
    result = img.copy()

    result = BilaterialPass(result, v2, 0, sigmaD, sigmaR)
    result = BilaterialPass(result, v2, 1, sigmaD, sigmaR)

    return result


def gaussian(sigma, pos):
    return (1.0 / np.sqrt(2.0 * pi * sigma * sigma)) * np.exp(-(pos * pos) / (2.0 * sigma * sigma))


def printI(img):
    plt.imshow(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(rgb)


def printI2(i1, i2):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cv2.cvtColor(i1, cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(cv2.cvtColor(i2, cv2.COLOR_BGR2RGB))


def pixelate(img, w, h):
    height, width = img.shape[:2]

    # Resize input to "pixelated" size
    temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Initialize output image
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)


def Inference(image_path, pixelSize, K, sigmaD, sigmaR, QuantizOutput, PixelOutput):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)

    height, width = img.shape[0], img.shape[1]
    h = height
    w = width

    S = Mystructure_tensor_2d(img, sigma, sigmaC)
    val, vec = eig_special_2d(S)
    v2 = vec.transpose(1, 2, 0)
    infCond = np.isinf(v2[:, :, 0]) | np.isinf(v2[:, :, 1])
    v2[infCond] = np.array([0, 1])

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)

    f = FlowBilaterial(img, v2, sigmaD, sigmaR)

    cv2_imshow(cv2.imread(image_path))

    # Return to RGB
    f = cv2.cvtColor(f, cv2.COLOR_YCR_CB2RGB)
    cv2_imshow(cv2.cvtColor(f, cv2.COLOR_RGB2BGR) * 255)

    # Quanitization
    if QuantizOutput:
        Z = f.reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        # Show Quanitization
        cv2_imshow(cv2.cvtColor(res2, cv2.COLOR_RGB2BGR) * 255)

    # Pixelization
    if PixelOutput:
        pkf = pixelate(res2, pixelSize, pixelSize)
        # Show Pixelization
        cv2_imshow(cv2.cvtColor(pkf, cv2.COLOR_RGB2BGR) * 255)