import glob
import pandas as pd
import os
import random
import shutil


def split_to_train_test():
    """splits 80/20 to folder"""
    data_path = "raw/"

    # path to destination folders
    train_folder = os.path.join(data_path, 'train')
    val_folder = os.path.join(data_path, 'test')

    # Define a list of image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Create a list of image filenames in 'data_path'
    imgs_list = [filename for filename in os.listdir(
        data_path) if os.path.splitext(filename)[-1] in image_extensions]

    # Sets the random seed
    random.seed(42)

    # Shuffle the list of image filenames
    random.shuffle(imgs_list)

    # determine the number of images for each set
    train_size = int(len(imgs_list) * 0.8)
    test_size = int(len(imgs_list) * 0.2)

    # Create destination folders if they don't exist
    for folder_path in [train_folder, val_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Copy image files to destination folders
    for i, f in enumerate(imgs_list):
        if i < train_size:
            dest_folder = train_folder
        elif i < train_size + test_size:
            dest_folder = val_folder
        shutil.copy(os.path.join(data_path, f), os.path.join(dest_folder, f))


def split_into_tag_folders(csv_path: str, file_path):
    """splits images from path into folders"""
    df = pd.read_csv(csv_path)
    for filename in glob.glob(f"{file_path}/*"):
        filename = filename.replace(f"{file_path}/", "")
        id_name = filename.split(".")[0]
        # print(id_name)
        row = df.iloc[[id_name]]
        e_tag = row.values[0][1]
        if not os.path.exists(f"{file_path}/{e_tag}"):
            os.makedirs(f"{file_path}/{e_tag}")
        shutil.move(f"{file_path}/{filename}",
                    f"{file_path}/{e_tag}/{filename}")


if __name__ == "__main__":
    split_into_tag_folders("./dataset.csv", "./raw/train")
