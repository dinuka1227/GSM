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

    # Calculate standard deviation for each quarter
    std_dev_top_left = np.std(top_left_quarter)
    std_dev_top_right = np.std(top_right_quarter)
    std_dev_bottom_left = np.std(bottom_left_quarter)
    std_dev_bottom_right = np.std(bottom_right_quarter)

    # Display the standard deviation values
    print(
        f"Standard Deviation - Top-left: {std_dev_top_left}, Top-right: {std_dev_top_right}, "
        f"Bottom-left: {std_dev_bottom_left}, Bottom-right: {std_dev_bottom_right}"
    )

    # Determine the quarter with the lowest standard deviation
    quarters_std_dev = [
        std_dev_top_left,
        std_dev_top_right,
        std_dev_bottom_left,
        std_dev_bottom_right,
    ]
    max_std_dev_index = np.argmax(quarters_std_dev)

    # Display the image with the highest contrast
    evenly_distributed_image = [
        top_left_quarter,
        top_right_quarter,
        bottom_left_quarter,
        bottom_right_quarter,
    ][max_std_dev_index]

    # Display images using Matplotlib
    plt.figure(figsize=(16, 4))

    plt.subplot(4, 2, 1)
    plt.imshow(top_left_quarter, cmap='gray')
    plt.title('Top Left Image')
    plt.axis('off')
    
    # Calculate and plot the histogram of the evnly distributed image
    plt.subplot(4, 2, 2)
    plt.hist(
        top_left_quarter.flatten(), bins=256, range=[0, 256], color="gray", alpha=0.7
    )
    plt.title("Top Left Image Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    plt.subplot(4, 2, 3)
    plt.imshow(top_right_quarter, cmap='gray')
    plt.title('Top Right Image')
    plt.axis('off')
    
    # Calculate and plot the histogram of the evnly distributed image
    plt.subplot(4, 2, 4)
    plt.hist(
        top_right_quarter.flatten(), bins=256, range=[0, 256], color="gray", alpha=0.7
    )
    plt.title("Top Right Image Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    plt.subplot(4, 2, 5)
    plt.imshow(bottom_left_quarter, cmap='gray')
    plt.title('Bottom Left Image')
    plt.axis('off')
    
    # Calculate and plot the histogram of the evnly distributed image
    plt.subplot(4, 2, 6)
    plt.hist(
        bottom_left_quarter.flatten(), bins=256, range=[0, 256], color="gray", alpha=0.7
    )
    plt.title("Bottom Left Image Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    plt.subplot(4, 2, 7)
    plt.imshow(bottom_right_quarter, cmap='gray')
    plt.title('Bottom Right Image')
    plt.axis('off')
    
    # Calculate and plot the histogram of the evnly distributed image
    plt.subplot(4, 2, 8)
    plt.hist(
        bottom_right_quarter.flatten(), bins=256, range=[0, 256], color="gray", alpha=0.7
    )
    plt.title("Bottom Right Image Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.show()

else:
    print(f"Error: Unable to load the image from {image_path}")
