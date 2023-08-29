# running as docker?
docker_running = False

# define repo path and add it to the path
from pathlib import Path
import sys, os
if not docker_running: # if we are running locally
    repo_path= Path.cwd().resolve()
    while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
        repo_path = repo_path.parent #go up one level
else: # if running in the container
    repo_path = Path('opt/usuari')
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
    stopping = None # for debugging

    ### 1. Settings
    # Expansion HP
    x_expansion = 865
    y_expansion = 865
    x_resizing = 512
    y_resizing = 512
    file_format = 'mha'
    folder_name = f'full-slice_{x_resizing}x{y_resizing}_all'
    data_directory = repo_path / 'data/challange_2023' / 'Test'
    data_directory.mkdir(exist_ok=True, parents=True)

    # transforms
    preprocess_im = Compose(
            [
                Resize((x_resizing, y_resizing), interpolation= InterpolationMode.BILINEAR),
            ]
    )

    # new images and labels
    save_dir = data_directory / folder_name
    save_dir.mkdir(exist_ok=True)
    im_dir = save_dir / f'image_{file_format}'
    im_dir.mkdir(exist_ok=True)

    ### 2. Create metadata
    raw_data = repo_path / 'input'
    files = [f for f in raw_data.glob('**/*') if f.suffix == '.nrrd']
    iter = tqdm(files, total=len(files))
    metadata = pd.DataFrame(columns=['case_id', 'data_path', 'shape'])
    for i, im_path in enumerate(iter):
        # get 3D image
        im_sitk = sitk.ReadImage(im_path)
        # store metadata
        metadata.loc[i, 'case_id'] = im_path.stem.split('_')[-1]
        metadata.loc[i, 'data_path'] = 'input/'+im_path.name
        metadata.loc[i, 'shape'] = im_sitk.GetSize()
    # save metadata
    metadata = metadata.sort_values(by=['case_id'])
    metadata.to_csv(data_directory / 'metadata.csv', index=False)
    print('Metadata saved')
    
    ### 3. Create 2D images
    metadata = pd.read_csv(data_directory / 'metadata.csv')
    iter = tqdm(metadata.iterrows(), total=len(metadata))
    # get example image
    for i, row in metadata.iterrows():    
        image_path = repo_path /  row['data_path']
        # get image
        im = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

        # now, we complete the images and labels to the expansion variables
        if im.shape[2]<x_expansion:
            print('Expanding x dimension')
            im = np.concatenate((im, np.zeros((im.shape[0], im.shape[1], x_expansion-im.shape[2]), dtype=np.int8)), axis=2)

        if im.shape[1]<y_expansion:
            # print('Expanding y dimension')
            im = np.concatenate((im, np.zeros((im.shape[0], y_expansion-im.shape[1], im.shape[2]), dtype=np.int8)), axis=1)

        # all z values available
        z_values = np.array(range(im.shape[0]))
        for z in z_values:
            # preprocess image
            im_slice = Image.fromarray(im[z])
            im_slice = preprocess_im(im_slice)
            im_slice = np.asarray(im_slice)
            # put channel first and repeat in RGB
            im_slice = np.repeat(np.expand_dims(im_slice, axis=0), 3, axis=0)
        
            # saving path
            save_name = f'id_{row["case_id"]}_slice_{z}.{file_format}'
            # save image
            sitk.WriteImage(sitk.GetImageFromArray(im_slice), str(im_dir / save_name))
        
        iter.update(1)
        # emergency stop
        if i == stopping:
            break
    iter.close()
    print('2D images saved')

if __name__ == '__main__':
    main()
