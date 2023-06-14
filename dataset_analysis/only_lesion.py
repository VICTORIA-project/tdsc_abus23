#Add repo path to the system path
from pathlib import Path, PurePosixPath
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
import SimpleITK
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

def main():
    # define paths, get list of images and labels and split them into train and test
    root_path = repo_path / 'data/challange_2023/Train'

    # new images and labels
    save_dir = repo_path / 'data/challange_2023' / 'only_lesion'
    save_dir.mkdir(exist_ok=True)
    im_dir = save_dir / 'image_mha'
    im_dir.mkdir(exist_ok=True)
    label_dir = save_dir / 'label_mha'
    label_dir.mkdir(exist_ok=True)

    # get list of images using metadata
    metadata = pd.read_csv(root_path / 'labels.csv')

    iter = tqdm(metadata.iterrows(), total=len(metadata))
    # get example image
    for i, row in metadata.iterrows():    
        # image files, replace \\ with /
        image_path = root_path /  row['data_path'].replace('\\', '/')
        label_path = root_path / row['mask_path'].replace('\\', '/')
        # get image and label
        im = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(image_path))
        label = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(label_path))

        # z values valid lesion
        z_values = np.unique(np.where(label)[0])
        for z in z_values:
            # preprocess image
            im_slice = Image.fromarray(im[z])
            im_slice = preprocess_im(im_slice)
            im_slice = np.asarray(im_slice)
            # # put channel first and repeat
            im_slice = np.repeat(np.expand_dims(im_slice, axis=0), 3, axis=0)

            # preprocess label
            label_slice = Image.fromarray(label[z])
            label_slice = preprocess_label(label_slice)
            label_slice = np.asarray(label_slice)
            # check if there is still a lesion
            if not np.any(label_slice):
                continue
        
            # saving path
            save_name = f'id_{row["case_id"]}_slice_{z}_label_{row["label"]}.mha'
            # save image
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(im_slice), str(im_dir / save_name))
            # save label
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(label_slice), str(label_dir / save_name))
        
        iter.update(1)
        # emergency stop
        if i == stopping:
            break

if __name__ == '__main__':
    main()
            