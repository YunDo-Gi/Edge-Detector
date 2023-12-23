import numpy as np
from PIL import Image


def generate_gaussian_kernel(size, sigma: int | float):
    # Generate a (size x size) matrix
    kernel = np.zeros((size, size))

    # Calculate the value of (size - 1) / 2
    k = (size - 1) / 2

    # Calculate the value of 2 * (sigma ** 2)
    two_sigma_squared = 2 * (sigma ** 2)

    # Iterate over the kernel
    for x in range(-int(k), int(k) + 1):
        for y in range(-int(k), int(k) + 1):
            # Calculate the value of e ^ (-(x ^ 2 + y ^ 2) / 2 * sigma ^ 2)
            e = np.exp(-(x ** 2 + y ** 2) / two_sigma_squared)

            # Calculate the value of 1 / (2 * pi * sigma ^ 2)
            c = 1 / (np.pi * two_sigma_squared)

            # Calculate the value of the kernel at (x + k, y + k) as e * c
            kernel[int(x + k), int(y + k)] = e * c

    # Normalize the values in the kernel
    kernel = kernel / np.sum(kernel)

    return kernel

def gaussian_convolution(image, gaussian_filter):
    # Get dimensions of image and filter
    image_height, image_width = image.shape[:2]
    filter_size = gaussian_filter.shape[0]

    # Calculate padding size
    padding_size = filter_size // 2

    # Pad the image
    padded_image = np.pad(image, padding_size, mode='constant')

    # Prepare output array
    output = np.zeros_like(image)

    # Apply convolution
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest
            region = padded_image[i:i+filter_size, j:j+filter_size]
            # Apply the filter
            output[i, j] = np.sum(region * gaussian_filter)

    return output

image_path = "assets/images/images.jpg"

# Open the image file
image = Image.open(image_path)

# Convert the image to a numpy array
image = np.array(image)

# Perform operations on the image

# Generate a Gaussian filter
gaussian_filter = generate_gaussian_kernel(5, 1)

# Apply Gaussian convolution
smoothed_image = gaussian_convolution(image, gaussian_filter)

image = Image.fromarray(smoothed_image)
image.show()

# Close the image file
image.close()

