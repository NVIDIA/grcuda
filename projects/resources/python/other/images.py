# Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NECSTLab nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#  * Neither the name of Politecnico di Milano nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 08:55:18 2020

@author: alberto.parravicini
"""

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

sobel_filter_diameter = 3
sobel_filter_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_filter_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def show_image(image):
    plt.imshow(image, cmap="gray")
    plt.show()


def sobel_filter(image):
    out = np.zeros(image.shape)
    rows, cols = image.shape
    radius = sobel_filter_diameter // 2
        
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
                        gradient_x = sobel_filter_x[x + radius][y + radius]
                        gradient_y = sobel_filter_y[x + radius][y + radius]
                        sum_gradient_x += gray_value_neigh * gradient_x
                        sum_gradient_y += gray_value_neigh * gradient_y
            out[i, j] = np.sqrt(sum_gradient_x**2 + sum_gradient_y**2)
    return out


def gaussian_kernel(diameter, sigma):
    kernel = np.zeros((diameter, diameter))
    
    mean = diameter / 2
    sum_tmp = 0
    for x in range(diameter):
        for y in range(diameter):
            kernel[x, y] = np.exp(-0.5 * ((x - mean)**2 + (y - mean)**2) / sigma**2)
            sum_tmp += kernel[x, y]
    for x in range(diameter):
        for y in range(diameter):
            kernel[x, y] /= sum_tmp
    return kernel


def gaussian_blur(image, kernel):
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


def linear_blur(image, diameter=5):
    out = np.zeros(image.shape)
    rows, cols = image.shape
    
    # Blur radius;
    radius = diameter // 2
    filter_area = diameter**2
    
    # Flatten image and kernel;
    image_1d = image.reshape(-1)
    
    for i in range(rows):
        for j in range(cols):
            sum_tmp = 0
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    nx = x + i
                    ny = y + j
                    if (nx >= 0 and ny >= 0 and nx < rows and ny < cols):
                        sum_tmp += image_1d[nx * cols + ny]
            out[i, j] = sum_tmp / filter_area
    return out


def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def truncate(image, minimum=0, maximum=1):
    out = image.copy()
    out[out < minimum] = minimum
    out[out > maximum] = maximum
    return out

#%%
if __name__ == "__main__":
    face = scipy.misc.face(gray=True)
    # Normalize to [0, 1];
    face = np.array(face, dtype=float) / 255
    show_image(face)
    
    #%% Part 1: Small blur on medium frequencies;
    
    # Compute blur filter;
    blurred_small = linear_blur(face, 3)
    show_image(blurred_small)
    # Find edges;
    edges_small = normalize(sobel_filter(blurred_small))
    show_image(edges_small)
    
    #%% Part 2: High blur on low frequencies;
    
    kernel = gaussian_kernel(3, 10)
    blurred_large = gaussian_blur(face, kernel)
    show_image(blurred_large)
    # Find edges;
    edges_large = sobel_filter(blurred_large)
    # Extend mask;
    edges_large = normalize(edges_large) * 5
    edges_large[edges_large > 1] = 1
    show_image(edges_large)
    
    #%% Part 3: Sharpen image;
    kernel2 = gaussian_kernel(3, 10)
    unsharpen = gaussian_blur(face, kernel2)   
    amount = 0.5
    sharpened = truncate(face * (1  + amount) - unsharpen * amount)
    show_image(sharpened)
    
    #%% Part 4: Merge sharpened image and low frequencies;
    image2 = normalize(sharpened * edges_large + blurred_large * (1 - edges_large))
    show_image(image2)
    
    #%% Part 5: Merge image and medium frequencies;
    image3 = image2 * edges_small + blurred_small * (1 - edges_small)
    show_image(image3)


    
    
    
    
    