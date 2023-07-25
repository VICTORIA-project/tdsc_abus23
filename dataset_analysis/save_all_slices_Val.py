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


def main():
    # define paths, get list of images and labels and split them into train and test
    root_path = repo_path / 'data/challange_2023/Val'

    # new images and labels
    save_dir = repo_path / 'data/challange_2023' / 'Val-all_slices'
    save_dir.mkdir(exist_ok=True)
    im_dir = save_dir / f'image_{file_format}'
    im_dir.mkdir(exist_ok=True)
    label_dir = save_dir / f'label_{file_format}'
    label_dir.mkdir(exist_ok=True)

    # data path
    data_path = root_path / 'DATA'
    # get all files with ending nrrd
    files = [f for f in data_path.glob('**/*') if f.suffix == '.nrrd']

    iter = tqdm(files, total=len(files))
    # get example image
    for i, im_path in enumerate(iter):
        # get image and label
        im = sitk.GetArrayFromImage(sitk.ReadImage(im_path))
        # get name DATA_100.nrrd
        name = im_path.stem
        # extract only id
        id = int(name.split('_')[-1])
        
        for z in range(im.shape[0]):
            # preprocess image
            im_slice = Image.fromarray(im[z])
            im_slice = preprocess_im(im_slice)
            im_slice = np.asarray(im_slice)
            # put channel first and repeat in RGB
            im_slice = np.repeat(np.expand_dims(im_slice, axis=0), 3, axis=0)
        
            # saving path
            save_name = f'id_{id}_slice_{z}.{file_format}'
            # save image
            sitk.WriteImage(sitk.GetImageFromArray(im_slice), str(im_dir / save_name))
        
        iter.update(1)
        # emergency stop
        if i == stopping:
            break

if __name__ == '__main__':
    main()
            