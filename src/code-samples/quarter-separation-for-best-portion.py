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

    # Get the dimensions of the cropped image
    cropped_height, cropped_width, _ = image.shape

    # Save the cropped image if needed
    # cv2.imwrite("path/to/save/cropped/image.jpg", image)
    
    # Convert the cropped image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalize the histogram
    equalized_gray_image = cv2.equalizeHist(gray_image)

    # Divide the equalized image into 4 equal parts
    height_half = cropped_height // 2
    width_half = cropped_width // 2
    
    print(f"Half Image Dimensions - Height: {height_half}, Width: {width_half}")

    # Top-left quarter
    top_left_quarter = equalized_gray_image[:height_half, :width_half]

    # Top-right quarter
    top_right_quarter = equalized_gray_image[:height_half, width_half:]

    # Bottom-left quarter
    bottom_left_quarter = equalized_gray_image[height_half:, :width_half]

    # Bottom-right quarter
    bottom_right_quarter = equalized_gray_image[height_half:, width_half:]

    # Display the original, grayscale, equalized, and divided images using Matplotlib
    plt.figure(figsize=(16, 4))

    # Divided Images
    plt.subplot(2, 2, 1)
    plt.imshow(top_left_quarter, cmap='gray')
    plt.title('Top-left Quarter')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(top_right_quarter, cmap='gray')
    plt.title('Top-right Quarter')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(bottom_left_quarter, cmap='gray')
    plt.title('Bottom-left Quarter')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(bottom_right_quarter, cmap='gray')
    plt.title('Bottom-right Quarter')
    plt.axis('off')

    plt.show()

else:
    print(f"Error: Unable to load the image from {image_path}")