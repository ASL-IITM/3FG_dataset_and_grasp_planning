import cv2
import numpy as np

# Read the images
image1 = cv2.imread('image1.png')
image2 = cv2.imread('image2.png')


hsv_image = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

# Define the range for blue color in HSV
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

# Create a mask for the blue areas
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

orange_hue = 10
hsv_image[mask > 0, 0] = orange_hue

final_result = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# Ensure the images have the same size
if final_result.shape != image2.shape:
    raise ValueError("Images must have the same dimensions and number of channels")

# Compute the average of the two images
average_image = cv2.addWeighted(final_result, 0.5, image2, 0.5, 0)

# Save or display the result
cv2.imwrite('average_image.jpg', average_image)
cv2.imshow('Average Image', average_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
