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

def main():

    # HP
    high_threshold = 0.65
    # load seed
    seeds_dir = repo_path / 'experiments/inference/segmentation/data/predictions/full-size/high-probs_no-lcc_limed_0.98_top-hat_0.0001'
    probs_dir = repo_path / 'experiments/inference/segmentation/data/predictions/full-size/multi-model_probs'
    save_dir = repo_path / f'experiments/inference/segmentation/data/predictions/full-size/high-probs_no-lcc_limed_0.98_top-hat_0.0001_{high_threshold}'
    save_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(os.listdir(seeds_dir))
    
    for name in tqdm(files):
        # load probs
        seed_path = seeds_dir / name
        seed = sitk.ReadImage(str(seed_path))
        seed = sitk.GetArrayFromImage(seed)

        # get probability map
        probs_path = probs_dir / name
        probs = sitk.ReadImage(str(probs_path))
        probs = sitk.GetArrayFromImage(probs)

        # use high_threshold on prob map
        possible_pixels = probs>high_threshold
        possible_pixels = possible_pixels.astype(np.uint8)

        # Define the structuring element for connected component analysis
        structuring_element = generate_binary_structure(3, 1)  # 3x3x3 connectivity

        # Perform connected component labeling
        labeled_mask, _ = label(possible_pixels, structure=structuring_element)

        intersection = labeled_mask*seed
        ideal_group = np.max(intersection)

        # use ideal group
        mask = labeled_mask == ideal_group
        mask = mask.astype(np.uint8)
        # save
        mask = sitk.GetImageFromArray(mask)
        # write
        sitk.WriteImage(mask, str(save_dir / name))

if __name__ == '__main__':
    main()
