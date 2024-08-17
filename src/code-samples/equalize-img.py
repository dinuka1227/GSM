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
else:
    print(f"Error: Unable to load the image from {image_path}")
    
# Convert the cropped image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Equalize the histogram
equalized_gray_image = cv2.equalizeHist(gray_image)

# Display the images
plt.figure(figsize=(8, 4))

# Grayscale Image
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Calculate and plot the histogram of the grayscale image
plt.subplot(2, 2, 2)
plt.hist(gray_image.flatten(), bins=256, range=[0,256], color='gray', alpha=0.7)
plt.title('Grayscale Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Equalized Grayscale Image
plt.subplot(2, 2, 3)
plt.imshow(equalized_gray_image, cmap='gray')
plt.title('Equalized Grayscale Image')
plt.axis('off')

# Calculate and plot the histogram of the equalized grayscale image
plt.subplot(2, 2, 4)
plt.hist(equalized_gray_image.flatten(), bins=256, range=[0,256], color='gray', alpha=0.7)
plt.title('Equalized Grayscale Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.show()