from PIL import Image
import os

# Path to your directory containing images
input_dir = r'./tmp/masks'

# Output directory to save the square images
output_dir = r'./tmp/Squ'

# Desired size for the square images (adjust as needed)
desired_size = 1024  # Change this value to your preferred square size


def resize_image(input_path, output_path, size):
    try:
        img = Image.open(input_path)
        img.thumbnail((size, size), Image.ANTIALIAS)

        # Create a new image with a white background for non-square images
        new_img = Image.new("RGB", (size, size), (255, 255, 255))
        position = ((size - img.width) // 2, (size - img.height) // 2)
        new_img.paste(img, position)

        new_img.save(output_path)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the input directory
for root, dirs, files in os.walk(input_dir):
    for file in files:
        file_path = os.path.join(root, file)
        output_path = os.path.join(output_dir, file)
        resize_image(file_path, output_path, desired_size)

print("Square images created and saved in the output directory.")
