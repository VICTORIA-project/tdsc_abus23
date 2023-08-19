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

def main():
    # HP
    top_threshold = 0.975

    saving_dir_name = f'high-probs_no-lcc_limed_{top_threshold}'
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

        valid_pixels = probs>top_threshold
        # use valid pixels as mask
        mask = np.zeros_like(probs)
        mask[valid_pixels] = 1
        mask = mask.astype(np.uint8)

        mask = sitk.GetImageFromArray(mask)
        # save
        saving_path = saving_dir / name
        sitk.WriteImage(mask, str(saving_path))

if __name__ == '__main__':
    main()