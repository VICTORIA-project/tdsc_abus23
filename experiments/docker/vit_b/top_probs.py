# running as docker?
docker_running = False

# define repo path and add it to the path
from pathlib import Path
import os, sys
if not docker_running: # if we are running locally
    repo_path= Path.cwd().resolve()
    while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
        repo_path = repo_path.parent #go up one level
else: # if running in the container
    repo_path = Path('opt/usuari')
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

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
    top_hat = 0.0001

    saving_dir_name = f'high-probs_top-hat_{top_hat}'
    saving_dir = repo_path / 'data/challange_2023/Test' / saving_dir_name
    saving_dir.mkdir(parents=True, exist_ok=True)

    # load probability map files
    probs_dir = repo_path / 'data/challange_2023/Test/vitb_probs'
    files = sorted(os.listdir(probs_dir))

    for name in tqdm(files):
        # load probs
        probs_path = probs_dir / name
        probs = sitk.ReadImage(str(probs_path))
        probs = sitk.GetArrayFromImage(probs)

        # create mask
        mask = np.zeros_like(probs)
        
        # use top values as mask
        valid_pixels = probs>(np.max(probs)-top_hat)
        mask[valid_pixels] = 1
        mask = mask.astype(np.uint8)

        # get lcc
        mask = lcc(mask)
        
        # # check with man_2
        # control_mask_dir = repo_path / 'experiments/inference/segmentation/data/predictions/full-size/man_2'
        # control_mask_path = control_mask_dir / name
        # control_mask = sitk.ReadImage(str(control_mask_path))
        # control_mask = sitk.GetArrayFromImage(control_mask)
        # # check if at least they share one pixel
        # if np.sum(mask*control_mask) != 0:
        #     print(f'{name}: Touch!')
        # else:
        #     print(f'{name}: Not touch! <----------------------------')

        mask = sitk.GetImageFromArray(mask)
        # save
        saving_path = saving_dir / name
        sitk.WriteImage(mask, str(saving_path))

if __name__ == '__main__':
    main()