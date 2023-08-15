#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import numpy as np
import SimpleITK as sitk
import pandas as pd
from torchvision.transforms import (
    Compose,
    Resize,
    InterpolationMode,
)
from PIL import Image
from tqdm import tqdm

def main():
    stopping = 2 # for debugging

    # Expansion HP
    x_expansion = 865
    y_expansion = 865
    x_resizing = 512
    y_resizing = 512
    file_format = 'mha'
    folder_name = f'full-slice_{x_resizing}x{y_resizing}'
    data_directory = repo_path / 'data/challange_2023' / 'Train'

    # transforms
    preprocess_im = Compose(
            [
                Resize((x_resizing, y_resizing), interpolation= InterpolationMode.BILINEAR),
            ]
    )

    preprocess_label = Compose(
            [
                Resize((x_resizing, y_resizing), interpolation= InterpolationMode.NEAREST), 
            ]
    )

    low_res_trans = Compose(
            [ 
                Resize((x_resizing//4, y_resizing//4), interpolation= InterpolationMode.NEAREST), 
            ]
    )


    # new images and labels
    save_dir = data_directory / folder_name
    save_dir.mkdir(exist_ok=True)
    im_dir = save_dir / f'image_{file_format}'
    im_dir.mkdir(exist_ok=True)
    label_dir = save_dir / f'label_{file_format}'
    label_dir.mkdir(exist_ok=True)

    # read metadata
    metadata = pd.read_csv(data_directory / 'extended_metadata.csv')
    
    iter = tqdm(metadata.iterrows(), total=len(metadata))
    # get example image
    for i, row in metadata.iterrows():    
        image_path = data_directory /  row['data_path']
        label_path = data_directory / row['mask_path']
        # get image and label
        im = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))

        # now, we complete the images and labels to the expansion variables
        if im.shape[2]<x_expansion:
            print('Expanding x dimension')
            im = np.concatenate((im, np.zeros((im.shape[0], im.shape[1], x_expansion-im.shape[2]), dtype=np.int8)), axis=2)

        if im.shape[1]<y_expansion:
            print('Expanding y dimension')
            im = np.concatenate((im, np.zeros((im.shape[0], y_expansion-im.shape[1], im.shape[2]), dtype=np.int8)), axis=1)

        # z values wıth lesion
        z_values = np.unique(np.where(label)[0])
        for z in z_values:
            # preprocess image
            im_slice = Image.fromarray(im[z])
            im_slice = preprocess_im(im_slice)
            im_slice = np.asarray(im_slice)
            # put channel first and repeat in RGB
            im_slice = np.repeat(np.expand_dims(im_slice, axis=0), 3, axis=0)

            # preprocess label
            label_slice = Image.fromarray(label[z])
            label_slice = preprocess_label(label_slice)
            # low resolution is needed for checking if there is still a lesion
            low_label_slice = low_res_trans(label_slice)
            low_label_slice = np.asarray(low_label_slice)
            label_slice = np.asarray(label_slice)
            # check if there is still a lesion
            if not np.any(label_slice): # it could disappear after preprocessing
                print(f'Nothing in the label. id; {row["case_id"]}, slice: {z}')
                continue
            if not np.any(low_label_slice): # it could disappear after low resolution
                print(f'Nothing in the low label. id; {row["case_id"]}, slice: {z}')
                continue
        
            # saving path
            save_name = f'id_{row["case_id"]}_slice_{z}_label_{row["label"]}.{file_format}'
            # save image
            sitk.WriteImage(sitk.GetImageFromArray(im_slice), str(im_dir / save_name))
            # save label
            sitk.WriteImage(sitk.GetImageFromArray(label_slice), str(label_dir / save_name))
        
        iter.update(1)
        # emergency stop
        if i == stopping:
            break

if __name__ == '__main__':
    main()
