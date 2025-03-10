import os

# Define the directory
directory = './tmp/masks'

# Get the list of files in the directory
files = sorted(os.listdir(directory))

# Loop over the files and rename them
for i, filename in enumerate(files):
    # Construct old file path
    old_file_path = os.path.join(directory, filename)

    # Construct new file name and path
    new_filename = f"{i + 1}{os.path.splitext(filename)[1]}"
    new_file_path = os.path.join(directory, new_filename)

    # Rename the file
    os.rename(old_file_path, new_file_path)

print("Files have been renamed successfully.")
