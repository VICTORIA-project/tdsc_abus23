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

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label, generate_binary_structure
from tqdm import tqdm

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
    top_buffer = 0.01
    normal_dir = repo_path / 'experiments/inference/segmentation/data/predictions/full-size/multi-model' # normal masks
    saving_dir_name = 'high-probs_lcc'
    saving_dir = repo_path / 'experiments/inference/segmentation/data/predictions/full-size' / saving_dir_name
    saving_dir.mkdir(parents=True, exist_ok=True)


    # load probability map files
    probs_dir = repo_path / 'experiments/inference/segmentation/data/predictions/full-size/multi-model_probs'
    files = sorted(os.listdir(probs_dir))

    for name in tqdm(files):
        # load mask
        mask_path = probs_dir / name
        mask = sitk.ReadImage(str(mask_path))
        mask = sitk.GetArrayFromImage(mask)
        
        # get max probability value in the mask
        max_prob = np.max(mask)
        threshold = max_prob - top_buffer
        binary_high_mask = mask > threshold
        binary_lcc = binary_high_mask.astype(np.uint8)
        # create lcc of high mask
        binary_lcc = lcc(binary_lcc)

        
        # read normal mask
        normal_mask = sitk.ReadImage(str(normal_dir / name))
        normal_mask = sitk.GetArrayFromImage(normal_mask)

        # Define the structuring element and perform connected component labeling
        structuring_element = generate_binary_structure(3, 1)  # 3x3x3 connectivity
        labeled_mask, _ = label(normal_mask, structure=structuring_element)

        # use binary lcc to find the optimal label
        high_lcc_mask = binary_lcc.astype(bool)*labeled_mask
        optimal_label = np.max(high_lcc_mask)
        final_mask = labeled_mask == optimal_label # get the optimal label

        # turn all non-zero values to 1
        final_mask = final_mask.astype(bool).astype(np.uint8)
        # save final mask
        final_mask = sitk.GetImageFromArray(final_mask)
        sitk.WriteImage(final_mask, saving_dir / name)

if __name__ == "__main__":
    main()