import numpy as np

# Steps of Canny Edge Detection Algorithm
# 1. Grey Scale Conversion
# 2. Noise Reduction - Gaussian Filter
# 3. Gradient Calculation
# 4. Non-maximum Suppression
# 5. Double Threshold
# 6. Edge Tracking by Hysteresis

#1. Grey Scale Conversion

def grayscale_conversion(image: np.ndarray):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Create a new empty grayscale image
    grayscale_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel in the image
    for i in range(height):
        for j in range(width):
            # Get the RGB values of the pixel
            r, g, b = image[i, j]

            # Calculate the grayscale value using the formula: Y = 0.299*R + 0.587*G + 0.114*B
            grayscale_value = int(0.299 * r + 0.587 * g + 0.114 * b)

            # Set the grayscale value in the new image
            grayscale_image[i, j] = grayscale_value

    return grayscale_image

#2. Noise Reduction - Gaussian Filter

def gaussian_filter(image: np.ndarray):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Create a new empty image
    new_image = np.zeros((height, width), dtype=np.uint8)

    # Define the Gaussian filter
    gaussian_filter = np.array(generate_gaussian_kernel(3, 5))

    # Get the dimensions of the filter
    filter_size = gaussian_filter.shape[0]

    # Calculate the padding size
    padding_size = int((filter_size - 1) / 2)

    # Iterate over each pixel in the image ignoring the border pixels
    for i in range(padding_size, height - padding_size):
        for j in range(padding_size, width - padding_size):
            # Calculate the sum of the products of the filter and the corresponding image pixels
            sum = 0
            for k in range(filter_size):
                for l in range(filter_size):
                    sum += gaussian_filter[k, l] * image[i + k - padding_size, j + l - padding_size]

            # Normalize the sum and set it as the corresponding pixel value in the new image
            new_image[i, j] = int(sum / 16)

    return new_image

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

#3. Gradient Calculation

def sobel_filters(image: np.ndarray):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Create a new empty image
    new_image = np.zeros((height, width), dtype=np.uint8)
    gradient_direction = np.zeros((height, width), dtype=np.float32)

    # Define the Sobel   filters
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Get the dimensions of the filter
    filter_size = sobel_x.shape[0]

    # Calculate the padding size
    padding_size = int((filter_size - 1) / 2)

    # Iterate over each pixel in the image ignoring the border pixels
    for i in range(padding_size, height - padding_size):
        for j in range(padding_size, width - padding_size):
            # Calculate the sum of the products of the filter and the corresponding image pixels
            sum_x = 0
            sum_y = 0
            for k in range(filter_size):
                for l in range(filter_size):
                    sum_x += sobel_x[k, l] * image[i + k - padding_size, j + l - padding_size]
                    sum_y += sobel_y[k, l] * image[i + k - padding_size, j + l - padding_size]

            # Set the corresponding pixel value in the new image
            new_image[i, j] = int(np.sqrt(sum_x ** 2 + sum_y ** 2))

            # Calculate the gradient direction using arctan2
            gradient_direction[i, j] = np.arctan2(sum_y, sum_x)

    return new_image, gradient_direction

#4. Non-maximum Suppression

def non_maximum_suppression(image: np.ndarray, gradient_direction: np.ndarray):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Create a new empty image
    new_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel in the image ignoring the border pixels
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Get the pixel value
            pixel = image[i, j]

            # Get the gradient direction
            direction = gradient_direction[i, j]

            # Set the neighbor pixels
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                left_pixel = image[i, j - 1]
                right_pixel = image[i, j + 1]
            elif (22.5 <= direction < 67.5):
                left_pixel = image[i + 1, j - 1]
                right_pixel = image[i - 1, j + 1]
            elif (67.5 <= direction < 112.5):
                left_pixel = image[i - 1, j]
                right_pixel = image[i + 1, j]
            elif (112.5 <= direction < 157.5):
                left_pixel = image[i - 1, j - 1]
                right_pixel = image[i + 1, j + 1]

            # Set the pixel value to 0 if it is smaller than any of its neighbors
            if (pixel < left_pixel) or (pixel < right_pixel):
                new_image[i, j] = 0
            # Set the pixel value to 255 if it is greater than both of its neighbors
            elif (pixel > left_pixel) and (pixel > right_pixel):
                new_image[i, j] = 255

    return new_image

#5. Double Threshold

def double_threshold(image: np.ndarray, low_threshold_ratio=0.05, high_threshold_ratio=0.09):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the high and low threshold values
    high_threshold = image.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    # Create a new empty image
    new_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel in the image
    for i in range(height):
        for j in range(width):
            # Get the pixel value
            pixel = image[i, j]

            # Set the pixel value to 255 if it is greater than the high threshold value
            if pixel >= high_threshold:
                new_image[i, j] = 255
            # Set the pixel value to 0 if it is smaller than the low threshold value
            elif pixel < low_threshold:
                new_image[i, j] = 0
            # Set the pixel value to 255 if it is between the low and high threshold values and has a neighbor with a value of 255
            elif (low_threshold <= pixel < high_threshold) and (255 in image[i - 1:i + 2, j - 1:j + 2]):
                new_image[i, j] = 255

    return new_image

#6. Edge Tracking by Hysteresis

def edge_tracking(image: np.ndarray):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Create a new empty image
    new_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel in the image ignoring the border pixels
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Set the pixel value to 255 if it has a neighbor with a value of 255
            if 255 in image[i - 1:i + 2, j - 1:j + 2]:
                new_image[i, j] = 255

    return new_image

# Canny Edge Detection Algorithm

def canny_edge_detection(image: np.ndarray):
    # Convert the image to grayscale
    grayscale_image = grayscale_conversion(image)

    # Apply the Gaussian filter
    gaussian_image = gaussian_filter(grayscale_image)

    # Apply the Sobel filters
    sobel_image = sobel_filters(gaussian_image)

    # Apply non-maximum suppression
    non_maximum_image = non_maximum_suppression(sobel_image[0], sobel_image[1])

    # Apply double threshold
    double_threshold_image = double_threshold(non_maximum_image)

    # Apply edge tracking by hysteresis
    canny_image = edge_tracking(double_threshold_image)

    return canny_image

# Test the Canny Edge Detection Algorithm

