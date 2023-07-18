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
device = 0

from importlib import import_module
from sklearn.model_selection import KFold
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    EnsureTyped,
    EnsureChannelFirstd,
    Resized,
)
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score

# special imports
from datasets_utils.datasets import ABUS_dataset
sys.path.append(str(repo_path / 'SAMed')) if str(repo_path / 'SAMed') not in sys.path else None
from SAMed.segment_anything import sam_model_registry


def main():
    # get SAM model
    checkpoint_dir = repo_path / 'checkpoints'
    sam, _ = sam_model_registry['vit_b'](image_size=256,
                                        num_classes=3,
                                        checkpoint=str(checkpoint_dir / 'sam_vit_b_01ec64.pth'),
                                        pixel_mean=[0, 0, 0],
                                        pixel_std=[1, 1, 1])
    # load lora model
    pkg = import_module('sam_lora_image_encoder')
    model = pkg.LoRA_Sam(sam, 4)
    # load weighs
    # load_path = weights_path / best_epoch
    load_path = repo_path / 'experiments/SAMed_ABUS/results/vanilla_3class/fold0/weights/epoch_73.pth'
    model.load_lora_parameters(str(load_path))
    model.eval()
    model.to(device)

    