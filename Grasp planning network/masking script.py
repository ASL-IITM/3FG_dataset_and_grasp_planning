import cv2
import numpy as np
import os

def mask_blue_objects(color_image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Define the range for blue color in HSV space
    lower_blue = np.array([110, 220, 160])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for blue color
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
     # Apply morphological operations to smooth the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Erosion followed by dilation
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Dilation followed by erosion

    # Highlight the blue objects in the image
    highlighted = cv2.bitwise_and(color_image, color_image, mask=mask)
   
    return mask, highlighted

def apply_mask_to_depth(mask, depth_image):
    # Apply the mask to the depth image
    masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)
    return masked_depth

def process_images_in_folder(color_folder, depth_folder, color_output_folder, depth_output_folder):
    # Create output folders if they don't exist
    if not os.path.exists(color_output_folder):
        os.makedirs(color_output_folder)
    if not os.path.exists(depth_output_folder):
        os.makedirs(depth_output_folder)

    # Get list of RGB files in the color folder
    color_files = [f for f in os.listdir(color_folder) if f.startswith('rgb_') and os.path.isfile(os.path.join(color_folder, f))]
    
    for color_file in color_files:
        # Construct full file paths
        color_image_path = os.path.join(color_folder, color_file)
        depth_image_name = color_file.replace('rgb_', 'depth_')
        depth_image_path = os.path.join(depth_folder, depth_image_name)

        # Check if corresponding depth image exists
        if not os.path.exists(depth_image_path):
            print(f"Depth image for {color_file} not found.")
            continue

        # Read the images
        color_image = cv2.imread(color_image_path)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # Ensure depth image is read correctly

        # Mask the blue objects in the color image
        mask, highlighted_color_image = mask_blue_objects(color_image)

        # Apply the mask to the depth image
        masked_depth_image = apply_mask_to_depth(mask, depth_image)

        # Save the processed images
        highlighted_color_output_path = os.path.join(color_output_folder, f"highlighted_{color_file}")
        masked_depth_output_path = os.path.join(depth_output_folder, f"masked_{depth_image_name}")

        cv2.imwrite(highlighted_color_output_path, highlighted_color_image)
        cv2.imwrite(masked_depth_output_path, masked_depth_image)

        print(f"Processed {color_file} and saved the results.")

def main():
    color_folder = 'D:\\Grasp planning\\egad_eval_set_0.12\\rgb'  # Update this with the path to your color images folder
    depth_folder = 'D:\\Grasp planning\\egad_eval_set_0.12\\depth'  # Update this with the path to your depth images folder
    color_output_folder = 'D:\\Grasp planning\\egad_eval_set_0.12\\masked_rgb'  # Update this with the path to the color output folder
    depth_output_folder = 'D:\\Grasp planning\\egad_eval_set_0.12\\masked_depth'  # Update this with the path to the depth output folder

    process_images_in_folder(color_folder, depth_folder, color_output_folder, depth_output_folder)

if __name__ == "__main__":
    main()


 
