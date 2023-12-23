import canny as canny
from PIL import Image
import numpy as np

image_path = "assets/images/image2.jpg"

# Open the image file
image = Image.open(image_path)

# Convert the image to a numpy array
image = np.array(image)

# Display some information about the image
print("Image size:", image.size)

# Perform operations on the image
image = canny.grayscale_conversion(image)
image = canny.gaussian_filter(image)
# image = canny.sobel_filters(image)
# image = canny.non_maximum_suppression(image[0], image[1])
image = Image.fromarray(image)
image.show()

# Close the image file
image.close()