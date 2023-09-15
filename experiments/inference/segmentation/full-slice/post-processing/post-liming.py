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
    top_threshold = 0.85

    ref_dir = repo_path / 'experiments/inference/segmentation/data/predictions/full-size/high-probs_lcc'
    saving_dir_name = f'high-probs_lcc-limed_{top_threshold}'
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
        ref_mask_path = ref_dir / name
        ref_mask = sitk.ReadImage(str(ref_mask_path))
        ref_mask = sitk.GetArrayFromImage(ref_mask)

        valid_pixels = probs>top_threshold
        ref_mask[~valid_pixels] = 0
        # create lcc of high mask
        ref_mask = lcc(ref_mask)
        ref_mask = sitk.GetImageFromArray(ref_mask)
        # save
        saving_path = saving_dir / name
        sitk.WriteImage(ref_mask, str(saving_path))

if __name__ == '__main__':
    main()