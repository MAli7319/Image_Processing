# Name: Muhammet Ali                      Student ID: 21993073
# Surname: ŞENTÜRK                        Lecture: AIN432 - Fundamentals of Image Processing Lab.
# Assignment 1

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.ndimage import gaussian_filter, median_filter

#######################################
#### Enter this parameters and run ####
##### No need to do anything else #####
######################## PARAMETERS ########################
input_img_folder_path = "data/"     # Path to the input images
output_img_folder_path = ""  # Path for the output images will be saved
FILTER = gaussian_filter                      # Set filter type
SIGMA = 0.5                                   # Set sigma parameter to blur the input image
K = 1.1                                       # Set K parameter to decide how much sigma will higher than the other kernel
LOW_SIGMA = 1                                 # Set sigma parameter to the lower sigma kernel
HIGH_SIGMA = LOW_SIGMA * K                    # Definition of the high sigma kernel
EPSILON = 10                                  # Set epsilon to threshold the pixels to get the binary image
NUMBER_OF_BITS = 16                           # Set number of bits to compress the image and reduce the number of colors
############################################################


# This function takes the channel array and quantizes it to given number of bits
def quantization(channel, number_of_bits):
    n_bits = np.linspace(0, channel.max(), number_of_bits)
    digit_img = np.digitize(channel, n_bits)
    quantized = (np.vectorize(n_bits.tolist().__getitem__)(digit_img - 1).astype(int))

    return quantized

# This function takes image input and the parameters to output cartoon-like image
def hsv(image, filter_, sigma, k, low_sigma, epsilon, n_of_bits):
    h_img = image[:, :, 0]   # Seperating the image channels
    s_img = image[:, :, 1]
    v_img = image[:, :, 2]

    if filter_ == gaussian_filter:   # Check if the filter is gaussian or not
        v_smooth = gaussian_filter(v_img, sigma=sigma)
    else:
        v_smooth = median_filter(v_img, size=3)

    smooth_img = np.dstack((h_img, s_img, v_smooth))   # Smooth image obtained

    gray_smooth_img = smooth_img[:, :, 2]   # Take only V channel to obtain grayscale image
    low_filter = gaussian_filter(gray_smooth_img, sigma=low_sigma)   # Definition of low filter
    high_filter = gaussian_filter(gray_smooth_img, sigma=low_sigma * k)   # Definition of high filter
    DoG_kernel = low_filter - high_filter                    # DoG kernel obtained
    threshold_img = (DoG_kernel > epsilon).astype(int)       # Thresholding the image wrt epsilon

    v_quantized = quantization(v_smooth, n_of_bits)          # Quantized image obtained by calling the corresponding function

    v_inverse = np.reciprocal(threshold_img + 1)             # Take the inverse of the thresholded image
    v_combine = np.multiply(v_inverse, v_quantized).astype("uint8")  # Inverse of thresholded and quantized are combined

    result_img = np.dstack((h_img, s_img, v_combine))   # Result image obtained

    return cv2.cvtColor(result_img, cv2.COLOR_HSV2RGB)   # Return the image with RGB channel


# Read the images one by one, call the hsv function, save the result image to the output path
for image_name in os.listdir(input_img_folder_path):
    image_path = os.path.join(input_img_folder_path, image_name)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2HSV)
    result = hsv(image, FILTER, SIGMA, K, LOW_SIGMA, EPSILON, NUMBER_OF_BITS)
    output_path = os.path.join(output_img_folder_path, image_name)

    print(image_name)
    plt.imshow(result)
    plt.axis("off")
    plt.savefig(output_path, dpi=220, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


# There are some functions used for getting comparison plots and step-by-step outputs.
# They can be complicated and involve code repeats.
# Since the main focus of this assignment is cartoon-like image implementation, I thought these are not important that
# much. So I have closed them to give simple look. If you wonder, you can check.

def read_image(path, color_space):
    img_name = path.split("/")[-1][:-4]

    if color_space == "rgb":
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        cartoonish_rgb(image, img_name, color_space, FILTER, SIGMA, K, LOW_SIGMA, EPSILON, NUMBER_OF_BITS)

    elif color_space == "hsv":
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)
        cartoonish_hsv(image, img_name, color_space, FILTER, SIGMA, K, LOW_SIGMA, EPSILON, NUMBER_OF_BITS)

    elif color_space == "lab":
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2LAB)
        cartoonish_lab(image, img_name, color_space, FILTER, SIGMA, K, LOW_SIGMA, EPSILON, NUMBER_OF_BITS)

    else:
        return
def plot_steps(img_name, color_space, smooth_img, edge_img, quantized_img, result_img):
    fig, ax = plt.subplots(2, 2, figsize=(15, 7))

    ax[0, 0].imshow(smooth_img)
    ax[0, 0].set_title("Step 1) Smoothed Image")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(edge_img, cmap="gray")
    ax[0, 1].set_title("Step 2) Edge Detected Image")
    ax[0, 1].axis("off")

    ax[1, 0].imshow(quantized_img)
    ax[1, 0].set_title("Step 3) Quantized Image")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(result_img)
    ax[1, 1].set_title("Step 4) Result Image")
    ax[1, 1].axis("off")

    plt.savefig(f"{img_name}_steps_in_{color_space}_space.png", bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()
def cartoonish_rgb(image, img_name, color_space, filter_, sigma, k, low_sigma, epsilon, n_of_bits):
    filter_name = filter_.__name__[:3]
    r_img = image[:, :, 0]
    g_img = image[:, :, 1]
    b_img = image[:, :, 2]

    r_smooth = filter_(r_img, sigma=sigma)
    g_smooth = filter_(g_img, sigma=sigma)
    b_smooth = filter_(b_img, sigma=sigma)
    smooth_img = np.dstack((r_smooth, g_smooth, b_smooth))

    gray_smooth_img = cv2.cvtColor(smooth_img, cv2.COLOR_RGB2GRAY)
    low_filter = gaussian_filter(gray_smooth_img, sigma=low_sigma)
    high_filter = gaussian_filter(gray_smooth_img, sigma=low_sigma * k)
    DoG_kernel = low_filter - high_filter
    threshold_img = (DoG_kernel > epsilon).astype(int)

    r_quantized = quantization(r_smooth, n_of_bits)
    g_quantized = quantization(g_smooth, n_of_bits)
    b_quantized = quantization(b_smooth, n_of_bits)
    quantized_img = np.dstack((r_quantized, g_quantized, b_quantized))

    r_inverse = np.reciprocal(threshold_img + 1)
    r_combine = np.multiply(r_inverse, r_quantized)

    g_inverse = np.reciprocal(threshold_img + 1)
    g_combine = np.multiply(g_inverse, g_quantized)

    b_inverse = np.reciprocal(threshold_img + 1)
    b_combine = np.multiply(b_inverse, b_quantized)

    result_img = np.dstack((r_combine, g_combine, b_combine))

    plot_steps(img_name, color_space, smooth_img, threshold_img, quantized_img, result_img)
    plt.show()
def cartoonish_hsv(image, img_name, color_space, filter_, sigma, k, low_sigma, epsilon, n_of_bits):
    h_img = image[:, :, 0]
    s_img = image[:, :, 1]
    v_img = image[:, :, 2]

    v_smooth = filter_(v_img, sigma=sigma)
    smooth_img = np.dstack((h_img, s_img, v_smooth))

    gray_smooth_img = smooth_img[:, :, 2]
    low_filter = gaussian_filter(gray_smooth_img, sigma=low_sigma)
    high_filter = gaussian_filter(gray_smooth_img, sigma=low_sigma * k)
    DoG_kernel = low_filter - high_filter
    threshold_img = (DoG_kernel > epsilon).astype(int)

    v_quantized = quantization(v_smooth, n_of_bits)
    quantized_img = np.dstack((h_img, s_img, v_quantized.astype("uint8")))

    v_inverse = np.reciprocal(threshold_img + 1)
    v_combine = np.multiply(v_inverse, v_quantized).astype("uint8")

    result_img = np.dstack((h_img, s_img, v_combine))

    plot_steps(img_name, color_space, cv2.cvtColor(smooth_img, cv2.COLOR_HSV2RGB),
               threshold_img,
               cv2.cvtColor(quantized_img, cv2.COLOR_HSV2RGB),
               cv2.cvtColor(result_img, cv2.COLOR_HSV2RGB))
    plt.show()
    return result_img
def cartoonish_lab(image, img_name, color_space, filter_, sigma, k, low_sigma, epsilon, n_of_bits):
    l_img = image[:, :, 0]
    a_img = image[:, :, 1]
    b_img = image[:, :, 2]

    l_smooth = filter_(l_img, sigma=sigma)
    smooth_img = np.dstack((l_smooth, a_img, b_img))

    gray_smooth_img = smooth_img[:, :, 0]
    low_filter = gaussian_filter(gray_smooth_img, sigma=low_sigma)
    high_filter = gaussian_filter(gray_smooth_img, sigma=low_sigma * k)
    DoG_kernel = low_filter - high_filter
    threshold_img = (DoG_kernel > epsilon).astype(int)

    l_quantized = quantization(l_smooth, n_of_bits)
    quantized_img = np.dstack((l_quantized.astype("uint8"), a_img, b_img))

    l_inverse = np.reciprocal(threshold_img + 1)
    l_combine = np.multiply(l_inverse, l_quantized).astype("uint8")

    result_img = np.dstack((l_combine, a_img, b_img))

    plot_steps(img_name, color_space, cv2.cvtColor(smooth_img, cv2.COLOR_LAB2RGB),
               threshold_img,
               cv2.cvtColor(quantized_img, cv2.COLOR_LAB2RGB),
               cv2.cvtColor(result_img, cv2.COLOR_LAB2RGB))
    plt.show()
params = {"filter_type" : [gaussian_filter, median_filter],
           "sigma" : [0.2, 0.5, 1.5, 3.5],
           "k" : [1.01, 1.1, 2, 4],
           "low_sigma" : [0.5, 1, 2, 5],
           "epsilon" : [0.1, 1, 10, 100],
           "number_of_bits" : [4, 8, 16, 32]}
# image = cv2.cvtColor(cv2.imread(input_img_folder_path), cv2.COLOR_BGR2HSV)
# img1 = hsv(image, FILTER,
#                   SIGMA,
#                   K,
#                   LOW_SIGMA,
#                   EPSILON,
#                   params["number_of_bits"][0])
# img2 = hsv(image, FILTER,
#                   SIGMA,
#                   K,
#                   LOW_SIGMA,
#                   EPSILON,
#                   params["number_of_bits"][1])
# img3 = hsv(image, FILTER,
#                   SIGMA,
#                   K,
#                   LOW_SIGMA,
#                   EPSILON,
#                   params["number_of_bits"][2])
# img4 = hsv(image, FILTER,
#                   SIGMA,
#                   K,
#                   LOW_SIGMA,
#                   EPSILON,
#                   params["number_of_bits"][3])
def compare_parameters(img1, img2, img3, img4):
    fig, ax = plt.subplots(2, 2, figsize=(15, 7))
    ax[0, 0].imshow(img1)
    ax[0, 0].set_title("Number of Bits = 4")
    ax[0, 0].axis("off")
    ax[0, 1].imshow(img2)
    ax[0, 1].set_title("Number of Bits = 8")
    ax[0, 1].axis("off")
    ax[1, 0].imshow(img3)
    ax[1, 0].set_title("Number of Bits = 16")
    ax[1, 0].axis("off")
    ax[1, 1].imshow(img4)
    ax[1, 1].set_title("Number of Bits = 32")
    ax[1, 1].axis("off")

    plt.savefig(f"bit_number_effect.png", bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()
def compare_filters(img1, img2):
    fig, ax = plt.subplots(1, 2, figsize=(13, 13))

    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_HSV2RGB))
    ax[0].set_title("Gaussian Filter")
    ax[0].axis("off")

    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_HSV2RGB))
    ax[1].set_title("Median Filter")
    ax[1].axis("off")

    plt.savefig("filter_comparison.png", bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()
