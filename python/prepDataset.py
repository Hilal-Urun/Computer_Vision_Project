from shutil import copyfile

import pandas as pd
import os, tqdm
import numpy as np

def load_categories(root):
    categories_file = 'category.txt'
    categ_df = pd.read_csv(os.path.join(root, categories_file), sep='\t')
    categories = list(categ_df.name)
    # Add background as category 0
    categories.insert(0, 'background')
    return categories


def merge_info(root):
    bbox_df = pd.DataFrame()
    bbox_filename = 'bb_info.txt'
    for class_dir in tqdm(os.listdir(root)):
        class_dir_path = os.path.join(root, class_dir)
        if os.path.isdir(class_dir_path):
            df = pd.read_csv(os.path.join(
                class_dir_path, bbox_filename), delim_whitespace=True)

            df['category'] = int(class_dir)  # add a class column
            indices = np.unique(df['img'].tolist())
            np.random.shuffle(indices)

            bbox_df = bbox_df.append(df)

    print('\nDone!\n')
    return bbox_df


# %%

def convert_dataset(root, dest):
    img_path = os.path.join(dest, 'Images')

    for folder, _, files in tqdm(os.walk(root), desc='Copying files'):

        for file in files:

            if file.endswith('.jpg'):

                path_file = os.path.join(folder, file)
                if not os.path.exists(img_path):
                    os.makedirs(img_path)

                copyfile(path_file, os.path.join(img_path, file))
            elif file == 'category.txt':

                path_file = os.path.join(folder, file)
                if not os.path.exists(dest):
                    os.makedirs(dest)

                copyfile(path_file, os.path.join(dest, file))
            else:
                continue

#
# import pandas as pd
# import glob
#
# file_path = "SUB_UECFOOD100/*/bbox.csv"
# csv_files = glob.glob(file_path)
# merged_data = pd.DataFrame()
# for file in csv_files:
#     df = pd.read_csv(file)
#     merged_data = merged_data.append(df, ignore_index=True)
#
# merged_data.to_csv("UECFOOD100/bbox.csv", index=True)
