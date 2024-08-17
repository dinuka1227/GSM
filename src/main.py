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

    # Threshold to separate yarn from background
    threshold_value = 175
    _, binary_image = cv2.threshold(evenly_distributed_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image to identify the shapes of the yarn.
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the original image
    contour_image = cv2.cvtColor(evenly_distributed_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)

    # Initialize variables for yarn count and total contour area
    yarn_count = len(contours)
    total_contour_area = 0
    
    # Calculate contour area
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        total_contour_area += contour_area

    # Calculate yarn density (contour area normalized by image area)
    image_area = evenly_distributed_image.shape[0] * evenly_distributed_image.shape[1]
    yarn_density = total_contour_area / image_area

    # Print the results
    print(f"Yarn Count: {yarn_count}")
    print(f"Total Contour Area: {total_contour_area}")
    print(f"Yarn Density: {yarn_density}")
    print(f"150 GSM: {150 / 0.2708544921875}")
    print(f"350 GSM: {350 / 0.2708544921875}")

    # Display images using Matplotlib
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(evenly_distributed_image, cmap='gray')
    plt.title('Best Portion Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(contour_image, cmap='gray')
    plt.title('Contour Image')
    plt.axis('off')

    plt.show()

else:
    print(f"Error: Unable to load the image from {image_path}")
