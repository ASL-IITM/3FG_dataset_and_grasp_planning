import torch
import numpy as np
from PIL import Image
from Vanilla_CNN_models import ResNet_Position_Orientation_2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Define the Model Architecture
model = ResNet_Position_Orientation_2(num_outputs=1, freeze_weights=False).to(device)

# Step 2: Load the Saved State Dictionary
model_path = 'D:\\Grasp planning network\\masked_model.pth'  # or 'model_epoch_20.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set the model to evaluation mode

# Step 3: Define the transform (should match the transform used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match input size of CNN architecture if required
    transforms.ToTensor()
])

# Step 4: Load and preprocess the RGBD image
rgb_image_path = 'D:\\Grasp planning\\test_images\\rgb_9.png'
depth_image_path = 'D:\\Grasp planning\\test_images\\depth_9.png'

rgb_image = Image.open(rgb_image_path).convert('RGB')
depth_image = Image.open(depth_image_path).convert('L')  # Convert to grayscale

# Convert depth image to (0,1) range and then to uint8 format
depth_array = np.array(depth_image)
scaled_depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min() + 1e-8)
scaled_depth_array = (scaled_depth_array * 255).astype(np.uint8)  # Convert to uint8

# Create depth image from array
depth_image = Image.fromarray(scaled_depth_array)

# Apply the transforms
rgb_tensor = transform(rgb_image)
depth_tensor = transform(depth_image)

# Ensure depth_tensor has one channel
if depth_tensor.shape[0] != 1:
    depth_tensor = depth_tensor[0, :, :].unsqueeze(0)  # Keep only one channel and add channel dimension

# Stack the RGB and depth tensors
rgbd_tensor = torch.cat((rgb_tensor, depth_tensor), dim=0).unsqueeze(0)  # Add batch dimension

# Step 5: Pass the Input Through the Model
rgbd_tensor = rgbd_tensor.to(device)
with torch.no_grad():
    pos_1_outputs, pos_2_outputs, pos_3_outputs, centroid, orientation, d = model(rgbd_tensor)

# Convert the outputs to numpy (if needed) and print the results
pos_1_outputs = pos_1_outputs.squeeze().cpu().numpy()
pos_2_outputs = pos_2_outputs.squeeze().cpu().numpy()
pos_3_outputs = pos_3_outputs.squeeze().cpu().numpy()

print("Position 1 Outputs:", pos_1_outputs)
print("Position 2 Outputs:", pos_2_outputs)
print("Position 3 Outputs:", pos_3_outputs)
print("Centroid:", centroid)
print("Orientation:", orientation)
print("d:", d)

# Step 8: Visualize the Triangle in 3D Space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [pos_1_outputs[0], pos_2_outputs[0], pos_3_outputs[0], pos_1_outputs[0]]
y = [pos_1_outputs[1], pos_2_outputs[1], pos_3_outputs[1], pos_1_outputs[1]]
z = [pos_1_outputs[2], pos_2_outputs[2], pos_3_outputs[2], pos_1_outputs[2]]

ax.plot(x, y, z, marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([0, 0.3])
ax.set_ylim([0, 0.3])
ax.set_zlim([0, 0.3])

ax.set_box_aspect([1, 1, 1])

ax.set_title('Predicted Triangle in 3D Space')

plt.show()
