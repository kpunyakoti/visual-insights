from PIL import Image
import os

# Specify the directory containing your images
input_directory = 'data/nuScenes_CamFront'

# Specify the output directory for resized images
output_directory = 'data/nuScenes_CamFront_resized'

# Set the desired width and height for resizing
new_width = 50
new_height = 50

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through each file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Add more image extensions if needed
        # Construct the full file paths
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        # Open the image
        with Image.open(input_path) as img:
            # Resize the image
            resized_img = img.resize((new_width, new_height))

            # Save the resized image to the output directory
            resized_img.save(output_path)

print("Image resizing complete.")
