# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 08:39:27 2021

Implement the image processing pipeline using Python and OpenCV, 
and Python implementations of the CUDA kernels.
Used for debugging, and to visualize intermediate results.

@author: albyr
"""

from skimage.io import imread, imsave
from skimage.filters import gaussian, sobel, unsharp_mask
from skimage.color import rgb2gray
from skimage import data, img_as_float

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
import time

BW = False
KERNEL_SMALL = 0.1
KERNEL_LARGE = 2
KERNEL_UNSHARPEN = 0.7

KERNEL_SMALL_DIAMETER = 3
KERNEL_SMALL_VARIANCE = 0.1
KERNEL_LARGE_DIAMETER = 5
KERNEL_LARGE_VARIANCE = 10
KERNEL_UNSHARPEN_DIAMETER = 3
KERNEL_UNSHARPEN_VARIANCE = 5

SOBEL_FILTER_DIAMETER = 3
SOBEL_FILTER_X = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
SOBEL_FILTER_Y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def time_function(name: str=None) -> Callable:
    """
    Decorator that simplifies timing a function call;
    :param name: name of the function or computation to measure
    :return: the output of the wrapped function
    """
    def inner_func(func) -> Callable:
        def func_call(self, *args, **kwargs) -> object:
            start = time.time()
            result = func(self, *args, **kwargs)
            end = time.time()
            print(f"{name if name is not None else func.__name__} took {end - start} sec")
            return result
        return func_call
    return inner_func


def gaussian_kernel(diameter, sigma):
    kernel = np.zeros((diameter, diameter))
    mean = diameter / 2
    sum_tmp = 0
    for x in range(diameter):
        for y in range(diameter):
            kernel[x, y] = np.exp(-0.5 * ((x - mean) ** 2 + (y - mean) ** 2) / sigma ** 2)
            sum_tmp += kernel[x, y]
    for x in range(diameter):
        for y in range(diameter):
            kernel[x, y] /= sum_tmp
    return kernel


def gaussian_blur_py(image, kernel):
    out = np.zeros(image.shape)
    rows, cols = image.shape

    # Blur radius;
    diameter = kernel.shape[0]
    radius = diameter // 2

    # Flatten image and kernel;
    image_1d = image.reshape(-1)
    kernel_1d = kernel.reshape(-1)

    for i in range(rows):
        for j in range(cols):
            sum_tmp = 0
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    nx = x + i
                    ny = y + j
                    if (nx >= 0 and ny >= 0 and nx < rows and ny < cols):
                        sum_tmp += kernel_1d[(x + radius) * diameter + (y + radius)] * image_1d[nx * cols + ny]
            out[i, j] = sum_tmp
    return out


def sobel_filter_py(image):
    out = np.zeros(image.shape)
    rows, cols = image.shape
    radius = SOBEL_FILTER_DIAMETER // 2

    for i in range(rows):
        for j in range(cols):
            sum_gradient_x = 0
            sum_gradient_y = 0
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    nx = x + i
                    ny = y + j
                    if (nx >= 0 and ny >= 0 and nx < rows and ny < cols):
                        gray_value_neigh = image[nx, ny]
                        gradient_x = SOBEL_FILTER_X[x + radius][y + radius]
                        gradient_y = SOBEL_FILTER_Y[x + radius][y + radius]
                        sum_gradient_x += gray_value_neigh * gradient_x
                        sum_gradient_y += gray_value_neigh * gradient_y
            out[i, j] = np.sqrt(sum_gradient_x ** 2 + sum_gradient_y ** 2)
    return out


def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def truncate(image, minimum=0, maximum=1):
    out = image.copy()
    out[out < minimum] = minimum
    out[out > maximum] = maximum
    return out


def scurve(img):
    img_out = img.copy()
    lut_b = lambda x: 0.7 * (1 / (1 + np.exp((-x + 0.5) * 10))) + 0.3 if x < 0.5 else 1 / (1 + np.exp((-x + 0.5) * 10))
    lut_r = lambda x: 0.8 * (1 / (1 + np.exp((-x + 0.5) * 7))) + 0.2 if x < 0.5 else (1 / (1 + np.exp((-x + 0.5) * 7)))
    lut_g = lambda x: 0.8 * (1 / (1 + np.exp((-x + 0.5) * 10))) + 0.2 if x < 0.5 else  (1 / (1 + np.exp((-x + 0.5) * 9)))
    lut_g2 = lambda x: x**1.4
    lut_b2 = lambda x: x**1.6
    img_out[:, :, 0] = np.vectorize(lut_b)(img[:, :, 0])
    img_out[:, :, 1] = np.vectorize(lut_g)(img[:, :, 1])
    img_out[:, :, 2] = np.vectorize(lut_r)(img[:, :, 2])
    
    img_out[:, :, 1] = np.vectorize(lut_g2)(img_out[:, :, 1])
    img_out[:, :, 0] = np.vectorize(lut_b2)(img_out[:, :, 0])
    
    return img_out
# plt.plot(np.linspace(0,1,255), scurve(np.linspace(0,1,255)))
#%%
@time_function()
def pipeline_golden(img):
    multichannel = not BW
    # Part 1: Small blur on medium frequencies;
    blurred_small = gaussian(img, sigma=(KERNEL_SMALL, KERNEL_SMALL), multichannel=multichannel)
    edges_small = normalize(sobel(blurred_small))

    # Part 2: High blur on low frequencies;
    blurred_large = gaussian(img, sigma=(KERNEL_LARGE, KERNEL_LARGE), multichannel=multichannel)
    edges_large = sobel(blurred_large)
    # Extend mask to cover a larger area;
    edges_large = truncate(normalize(edges_large) * 5)

    # Part 3: Sharpen image;
    amount = 10
    sharpened = unsharp_mask(img, radius=KERNEL_UNSHARPEN, amount=amount, multichannel=multichannel)

    # Part 4: Merge sharpened image and low frequencies;
    image2 = normalize(sharpened * edges_large + blurred_large * (1 - edges_large))

    # Part 5: Merge image and medium frequencies;
    result = image2 * edges_small + blurred_small * (1 - edges_small)
    
    # Part 6: Apply LUT;
    result_lut = scurve(result)
    
    return result_lut, [blurred_small, edges_small, blurred_large, edges_large, sharpened, image2, result]


@time_function()
def pipeline_bw(img):
    
    # Create kernels for blur;
    kernel_small_cpu = gaussian_kernel(KERNEL_SMALL_DIAMETER, KERNEL_SMALL_VARIANCE)
    kernel_large_cpu = gaussian_kernel(KERNEL_LARGE_DIAMETER, KERNEL_LARGE_VARIANCE)
    kernel_unsharpen_cpu = gaussian_kernel(KERNEL_UNSHARPEN_DIAMETER, KERNEL_UNSHARPEN_VARIANCE)
    
    # Part 1: Small blur on medium frequencies;
    blurred_small = gaussian_blur_py(img, kernel_small_cpu)
    edges_small = normalize(sobel_filter_py(blurred_small))

    # Part 2: High blur on low frequencies;
    blurred_large = gaussian_blur_py(img, kernel_large_cpu)
    edges_large = sobel_filter_py(blurred_large)
    # Extend mask to cover a larger area;
    edges_large = truncate(normalize(edges_large) * 5)

    # Part 3: Sharpen image;
    unsharpen = gaussian_blur_py(img, kernel_unsharpen_cpu)
    amount = 8
    sharpened = truncate(img * (1 + amount) - unsharpen * amount)

    # Part 4: Merge sharpened image and low frequencies;
    image2 = normalize(sharpened * edges_large + blurred_large * (1 - edges_large))

    # Part 5: Merge image and medium frequencies;
    result = image2 * edges_small + blurred_small * (1 - edges_small)
    return result, [blurred_small, edges_small, blurred_large, edges_large, sharpened, image2]


if __name__ == "__main__":
        
    # img = imread("puppy.jpg")
    img = img_as_float(data.astronaut())
    if BW:
        img = rgb2gray(img) # Output is a [0,1] matrix;
    
    # Golden pipeline;
    result, other = pipeline_golden(img)
    
    fig, axes = plt.subplots(4, 2, figsize=(6, 6))
    ax = axes.ravel()
    
    cmap =  plt.cm.gray if BW else None
    ax[0].imshow(img, cmap=cmap)
    ax[1].imshow(other[0], cmap=cmap)
    ax[2].imshow(np.dot(other[1][...,:3], [0.33, 0.33, 0.33]), cmap='gray') # other[1], cmap=plt.cm.gray)
    ax[3].imshow(other[2], cmap=cmap)
    ax[4].imshow(np.dot(other[3][...,:3], [0.33, 0.33, 0.33]), cmap='gray')
    ax[5].imshow(other[4], cmap=cmap)
    ax[6].imshow(other[5], cmap=cmap)
    ax[7].imshow(result, cmap=cmap)
    for i in ax:
        i.axis("off")
    fig.tight_layout()
    plt.show()
    fig.savefig("astronaut_g.jpg")
       
    # Custom BW pipeline;
    result2 = np.zeros(img.shape)
    other2 = [np.zeros(img.shape) for i in range(len(other))]
    for i in range(img.shape[-1]):
        result2[:, :, i], tmp = pipeline_bw(img[:, :, i])
        for j, x in enumerate(tmp):
            other2[j][:, :, i] = x
    
    # fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    # ax = axes.ravel()
    
    cmap =  plt.cm.gray if BW else None
    ax[0].imshow(img, cmap=cmap)
    ax[1].imshow(other2[2], cmap=cmap)
    ax[2].imshow(other2[3], cmap=cmap)
    ax[3].imshow(result2, cmap=cmap)
    for i in ax:
        i.axis("off")
    fig.tight_layout()
    plt.show()
    fig.savefig("astronaut_py.jpg")
