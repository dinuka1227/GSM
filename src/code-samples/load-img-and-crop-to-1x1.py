import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the image file
image_path = "src/data/test-data/denim-0001.jpg"

# Load the image using OpenCV
image = cv2.imread(image_path)

# Check if the image is loaded successfully
if image is not None:
    # Get the dimensions of the image
    height, width, channels = image.shape

    print(f"Image Dimensions - Height: {height}, Width: {width}, Channels: {channels}")
    
    # Get the minimum of width and height
    min_dimension = min(height, width)
    
    print(f"Minimum Dimension: {min_dimension}")
    
    # Crop the image to min_dimension x min_dimension
    image = image[:min_dimension, :min_dimension, :]

    # Save the cropped image if needed
    # cv2.imwrite("path/to/save/cropped/image.jpg", image)
else:
    print(f"Error: Unable to load the image from {image_path}")
    
# Convert the cropped image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the images
plt.figure(figsize=(8, 4))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Grayscale Image
plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.show()