#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    InterpolationMode,
)
from PIL import Image

# HP
resolution = 256
stopping = None
file_format = 'mha'

# transforms
preprocess_im = Compose(
        [ # classic squared aspect-preserved centered image
            Resize(resolution, interpolation= InterpolationMode.BILINEAR),
            CenterCrop(resolution), 
        ]
)

preprocess_label = Compose(
        [ # classic squared aspect-preserved centered image
            Resize(resolution, interpolation= InterpolationMode.NEAREST),
            CenterCrop(resolution), 
        ]
)

low_res_trans = Compose(
        [ # classic squared aspect-preserved centered image
            Resize(64, interpolation= InterpolationMode.NEAREST),
            CenterCrop(64), 
        ]
)

def main():
    # define paths, get list of images and labels and split them into train and test
    root_path = repo_path / 'data/challange_2023/Train'

    # new images and labels
    save_dir = repo_path / 'data/challange_2023' / 'with_lesion'
    save_dir.mkdir(exist_ok=True)
    im_dir = save_dir / f'image_{file_format}'
    im_dir.mkdir(exist_ok=True)
    label_dir = save_dir / f'label_{file_format}'
    label_dir.mkdir(exist_ok=True)

    # get list of images using metadata
    metadata = pd.read_csv(root_path / 'extended_metadata.csv')

    iter = tqdm(metadata.iterrows(), total=len(metadata))
    # get example image
    for i, row in metadata.iterrows():    
        image_path = root_path /  row['data_path']
        label_path = root_path / row['mask_path']
        # get image and label
        im = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))

        # z values valid lesion
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
            