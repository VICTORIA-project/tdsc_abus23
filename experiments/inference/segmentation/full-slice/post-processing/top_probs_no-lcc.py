#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = 0

# top bueno: high-probs_no-lcc_limed_0.98_top-hat_0.0001

from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

from scipy.ndimage import label, generate_binary_structure

def lcc(mask:np.array):
    """generate largest connected component of a mask

    Args:
        mask (np.array): multi object mask

    Returns:
        np.array: array containing only the largest connected component
    """
    # Define the structuring element for connected component analysis
    structuring_element = generate_binary_structure(3, 1)  # 3x3x3 connectivity

    # Perform connected component labeling
    labeled_mask, _ = label(mask, structure=structuring_element)

    # Find the size of each connected component
    component_sizes = np.bincount(labeled_mask.ravel())

    # Identify the label of the largest component (excluding background)
    largest_component_label = np.argmax(component_sizes[1:]) + 1

    # Create a new mask containing only the largest component
    largest_component_mask = labeled_mask == largest_component_label
    # transform boolean to int
    largest_component_mask = largest_component_mask.astype(np.uint8)

    return largest_component_mask

def main():
    # HP
    top_threshold = 0.98
    top_values=True
    top_hat = 0.0001

    saving_dir_name = f'high-probs_no-lcc_limed_{top_threshold}' if not top_values else f'high-probs_no-lcc_limed_{top_threshold}_top-hat_{top_hat}'
    saving_dir = repo_path / 'experiments/inference/segmentation/data/predictions/full-size' / saving_dir_name
    saving_dir.mkdir(parents=True, exist_ok=True)

    # load probability map files
    probs_dir = repo_path / 'experiments/inference/segmentation/data/predictions/full-size/multi-model_probs'
    files = sorted(os.listdir(probs_dir))

    for name in tqdm(files):
        # load probs
        probs_path = probs_dir / name
        probs = sitk.ReadImage(str(probs_path))
        probs = sitk.GetArrayFromImage(probs)
        # apply threshold
        valid_pixels = probs>top_threshold

        # # check
        # valid_probs = probs[valid_pixels]
        # uniques = np.unique(valid_probs)
        # print(f'{name}: top {hat} values: {uniques[-10:]}')

        # create mask
        mask = np.zeros_like(probs)
        
        if top_values:
            # # use top 10 unique values as mask
            # for value in uniques[-number_top:]:
            #     mask[probs==value] = 1
            #     mask = mask.astype(np.uint8)
            # use top hat
            valid_pixels = probs>(np.max(probs)-top_hat)
            mask[valid_pixels] = 1
            mask = mask.astype(np.uint8)

        else:
            # use valid pixels as mask
            mask[valid_pixels] = 1
            mask = mask.astype(np.uint8)

        # get lcc
        mask = lcc(mask)
        
        # check with man_2
        control_mask_dir = repo_path / 'experiments/inference/segmentation/data/predictions/full-size/man_2'
        control_mask_path = control_mask_dir / name
        control_mask = sitk.ReadImage(str(control_mask_path))
        control_mask = sitk.GetArrayFromImage(control_mask)
        # check if at least they share one pixel
        if np.sum(mask*control_mask) != 0:
            print(f'{name}: Touch!')
        else:
            print(f'{name}: Not touch! <----------------------------')

        mask = sitk.GetImageFromArray(mask)
        # save
        saving_path = saving_dir / name
        sitk.WriteImage(mask, str(saving_path))

if __name__ == '__main__':
    main()