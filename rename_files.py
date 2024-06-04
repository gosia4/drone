import os
import re
def rename_files(directory):
    for filename in os.listdir(directory):
        if "frame_" in filename:
            new_filename = re.sub(r'frame_0*', 'frame_', filename)
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, new_filename)
            os.rename(src, dst)
            print(f"Renamed: {src} to {dst}")

# Replace 'your_directory_path' with the path to your directory
# rename_files(r"C:\Users\gosia\OneDrive - vus.hr\Desktop\New folder\drone\thermographic_data\validate\images\free_3")

import shutil


def rename_images(source_dir, target_dir, prefix="free_3_frame"):
    """
    Rename images in source_dir and save them sequentially in target_dir
    without changing their order.

    Parameters:
    - source_dir: Directory containing the original images.
    - target_dir: Directory where renamed images will be saved.
    - prefix: Prefix for the new image names.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # List all files in the source directory
    files = [f for f in sorted(os.listdir(source_dir)) if os.path.isfile(os.path.join(source_dir, f))]

    count = 0
    for file_name in files:
        # Define the new file name
        new_name = f"{prefix}_{count}.txt"
        # Define the source and target file paths
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, new_name)

        # Copy and rename the file
        shutil.copy2(source_file, target_file)

        print(f"Renamed {source_file} to {target_file}")
        count += 1


# Example usage
source_directory = r"C:\Users\gosia\OneDrive - vus.hr\Desktop\New folder\drone\thermographic_my_data_z_zerami\validate\labels\free_3"
target_directory = r"C:\Users\gosia\OneDrive - vus.hr\Desktop\New folder\drone\thermographic_data\validate\labels\free_3"
rename_images(source_directory, target_directory)
