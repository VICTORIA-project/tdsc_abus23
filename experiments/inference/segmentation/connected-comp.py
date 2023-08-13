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

from scipy.ndimage import label, generate_binary_structure
import SimpleITK as sitk
import numpy as np


def main():
    
    # get all files in the directory
    directory_path = repo_path / 'experiments/inference/segmentation/data/predictions/vanilla'
    all_files = [file for file in os.listdir(directory_path) if file.endswith('.nii.gz')]

    for path in all_files:
        name = Path(path).stem
        segmentation_mask = sitk.ReadImage(directory_path / path)
        segmentation_mask = sitk.GetArrayFromImage(segmentation_mask)
        print(f'The size of the segmentation mask is {segmentation_mask.shape}')

        # Define the structuring element for connected component analysis
        structuring_element = generate_binary_structure(3, 1)  # 3x3x3 connectivity

        # Perform connected component labeling
        labeled_mask, _ = label(segmentation_mask, structure=structuring_element)

        # Find the size of each connected component
        component_sizes = np.bincount(labeled_mask.ravel())

        # Identify the label of the largest component (excluding background)
        largest_component_label = np.argmax(component_sizes[1:]) + 1

        # Create a new mask containing only the largest component
        largest_component_mask = labeled_mask == largest_component_label
        # transform boolean to int
        largest_component_mask = largest_component_mask.astype(np.int8)

        # save largest component mask
        largest_component_mask = sitk.GetImageFromArray(largest_component_mask)
        saving_dir = repo_path / 'experiments/inference/segmentation/data/predictions/vanilla-lcc'
        sitk.WriteImage(largest_component_mask, saving_dir / f'MASK_{id}_largest_component.nii.gz')