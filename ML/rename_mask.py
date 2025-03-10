import os

# Define the directory where the files are located
directory = 'C:/Users/sahil/PycharmProjects/ISRO/ML/tmp/masks'

# Get a list of all files in the directory
files = os.listdir(directory)

# Filter out files that match the pattern "tile_x" where x is a digit
tile_files = [f for f in files if f.startswith('tile_') and f[5:-4].isdigit() and f.endswith('.jpg')]

# Sort the files by the numerical value following "tile_"
tile_files.sort(key=lambda f: int(f[5:-4]))

# Debug: Print the list of files to be renamed
print("Files to be renamed:", tile_files)

# Rename each file
for i, filename in enumerate(tile_files):
    # Generate the new name with .jpg extension
    new_name = f"{i + 1}.jpg"
    # Form the full old and new file paths
    old_file = os.path.join(directory, filename)
    new_file = os.path.join(directory, new_name)
    # Rename the file
    try:
        os.rename(old_file, new_file)
        print(f"Renamed {old_file} to {new_file}")
    except Exception as e:
        print(f"Failed to rename {old_file} to {new_file}: {e}")
