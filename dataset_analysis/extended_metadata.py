#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
import numpy as np


def main():

    # define root (data) path
    root_path = repo_path / 'data/challange_2023/Train'

    # get list of images using metadata
    metadata = pd.read_csv(root_path / 'labels.csv')
    # image files, replace \\ with / for windows
    metadata['data_path'] = metadata['data_path'].str.replace('\\', '/', regex=False)
    metadata['mask_path'] = metadata['mask_path'].str.replace('\\', '/', regex=False)
    
    # create extended metadata file
    extended_metadata = None
    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        # get path for reading
        image_path = root_path /  row['data_path']
        mask_path = root_path /  row['mask_path']
        im = sitk.ReadImage(image_path) # read
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)) # read
        # get shape
        row['shape'] = im.GetSize()
        # # get spacing
        # row['spacing'] = im.GetSpacing()
        # # get origin
        # row['origin'] = im.GetOrigin()
        # # get direction
        # row['direction'] = im.GetDirection()
        # # get pixel type
        # row['pixel_type'] = im.GetPixelIDTypeAsString()

        # mask info
        # z-slices where mask is not zero
        z_values = np.unique(np.where(mask)[0])
        row['slices_num'] = len(z_values)
        row['slice_pos'] = z_values[0], z_values[-1]

        # append to extended metadata
        extended_metadata = pd.concat([extended_metadata, row], axis=1)
    # save extended metadata
    extended_metadata = extended_metadata.T
    extended_metadata.to_csv(root_path / 'extended_metadata.csv', index=False)

if __name__ == '__main__':
    main()