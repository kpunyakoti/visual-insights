import tarfile
import time
import os

# Check if .tgz file exists under data directory path.
# For project demo purposes, it will check only for nuScenes_CamFront_resized.
data_directory_path = 'data'
os.chdir("../")
target_path = os.path.join(data_directory_path,'nuScenes_CamFront_resized')
if not os.path.exists(target_path):
    tgz_filename = 'nuScenes_CamFront_resized.tgz'
    tgz_filepath =  os.path.join(data_directory_path,tgz_filename)
    if os.path.exists(tgz_filepath):
        with tarfile.open(tgz_filepath, 'r:gz') as tar:
            tar.extractall(data_directory_path)
            time.sleep(10)
        print(f"{tgz_filename} untarred successfully.")
    else:
        print(f"Error: {tgz_filename} not found.")
else:
    print(f"{target_path} already exists.")