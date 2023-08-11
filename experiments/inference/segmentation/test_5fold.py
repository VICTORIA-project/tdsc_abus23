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
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    EnsureTyped,
    Resized,
)
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
import SimpleITK as sitk
from PIL import Image
import torchvision
import pandas as pd

# special imports
from datasets_utils.datasets import ABUS_test
sys.path.append(str(repo_path / 'SAMed')) if str(repo_path / 'SAMed') not in sys.path else None
from SAMed.segment_anything import sam_model_registry

import re
# Define a custom sorting key function
def slice_number(filename):
    match = re.search(r'slice_(\d+)\.mha', filename)
    if match:
        return int(match.group(1))
    return -1  # Default value if the pattern is not found

def main():
    # HP
    batch_size = 8
    num_classes = 1

    # get SAM model
    checkpoint_dir = repo_path / 'checkpoints'
    sam, _ = sam_model_registry['vit_b'](image_size=256,
                                        num_classes=num_classes,
                                        checkpoint=str(checkpoint_dir / 'sam_vit_b_01ec64.pth'),
                                        pixel_mean=[0, 0, 0],
                                        pixel_std=[1, 1, 1])
    # load lora model
    pkg = import_module('sam_lora_image_encoder')
    model = pkg.LoRA_Sam(sam, 4)
    
    # list of optimum weights per fold
    optimum_weights = [
        'experiments/SAMed_ABUS/results/scratch_c1_val-all/fold0/weights/epoch_19.pth',
        'experiments/SAMed_ABUS/results/scratch_c1_val-all/fold1/weights/epoch_9.pth',
        'experiments/SAMed_ABUS/results/scratch_c1_val-all/fold2/weights/epoch_3.pth',
        'experiments/SAMed_ABUS/results/scratch_c1_val-all/fold3/weights/epoch_41.pth',
        'experiments/SAMed_ABUS/results/scratch_c1_val-all/fold4/weights/epoch_3.pth'
    ]

    val_transform = Compose(
            [
                ScaleIntensityd(keys=["image"]),
                Resized(keys=["image"], spatial_size=(256, 256),mode=['area']),
                EnsureTyped(keys=["image"])
            ])
    

    metadata_path = repo_path / 'data/challange_2023/Val/metadata.csv'
    metadata = pd.read_csv(metadata_path)

    
    for pat_id in range(100,130,1): # each val id
        patient_meta = metadata[metadata['case_id'] == pat_id]
        original_shape = patient_meta['shape'].apply(lambda x: tuple(map(int, x[1:-1].split(',')))).values[0]
        diff_shape = original_shape[0]-original_shape[1]
        
        # get data
        root_path = repo_path / 'data/challange_2023/Val/all-slices'
        path_images = (root_path / "image_mha")
        # get all files in the folder in a list, only mha files
        image_files = sorted([file for file in os.listdir(path_images) if file.endswith('.mha')])
        # now, we will check if the path has at least one of the ids in the train_ids list
        val_files = [file for file in image_files if f'id_{pat_id}_' in file]
        val_files = sorted(val_files, key=slice_number)
        # create final paths
        image_files = np.array([path_images / i for i in val_files])
        # define dataset and dataloader
        db_val = ABUS_test(transform=val_transform,list_dir=image_files)   
        valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        print(f'The patient id is {pat_id}')
        print(f'The number of slices is {len(db_val)}')
        
        # store final mask per patient
        output_mask_final = torch.zeros((len(db_val),num_classes+1,256,256))
        for model_path in optimum_weights: # for each model learned
            # load weighs
            load_path = repo_path / model_path
            model.load_lora_parameters(str(load_path))
            model.eval()
            model.to(device)
            
            model_mask = []
            for sample_batch in valloader: # get some slides
                # get data
                image_batch = sample_batch["image"].to(device)
                # forward and losses computing
                outputs = model(image_batch, True, 256)
                # stack the masks
                model_mask.append(outputs['masks'].detach().cpu())
            # stack tensors in a single one
            model_mask = torch.cat(model_mask, dim=0)
            print(f'The shape of the output is {model_mask.shape}')
            output_mask_final += model_mask
        # get the mean
        output_mask_final /= len(optimum_weights)
        output_mask_final = torch.argmax(torch.softmax(output_mask_final, dim=1), dim=1, keepdim=True)
        # remove second dimension channel
        output_mask_final = output_mask_final[:,0,:,:]
        # save as nii.gz file
        output_mask_final = output_mask_final.cpu().numpy()
        # save as int8
        output_mask_final = output_mask_final.astype(np.int8)
        print(f'The shape of the final output is {output_mask_final.shape}')

        # reshape each slice
        resize_stack = []
        for slice_num in range(output_mask_final.shape[0]):
            im_slice = output_mask_final[slice_num,:,:]
            im_slice = Image.fromarray(im_slice)
            im_slice_comeback = torchvision.transforms.Resize(original_shape[1], interpolation= torchvision.transforms.InterpolationMode.NEAREST)(im_slice)
            padded = torchvision.transforms.Pad((int(diff_shape/2),0))(im_slice_comeback)
            padded = np.asanyarray(torchvision.transforms.Resize(original_shape[:2][::-1], interpolation= torchvision.transforms.InterpolationMode.NEAREST)(padded))
            resize_stack.append(padded)
        # stack all slices
        output_mask_final = np.stack(resize_stack, axis=0)
        print(f'The shape of the final output is {output_mask_final.shape}')

        saving_path = repo_path / 'experiments/inference/segmentation/data/predictions' / f'MASK_{pat_id}.nii.gz'
        # save the mask as nii.gz
        sitk.WriteImage(sitk.GetImageFromArray(output_mask_final), str(saving_path))



if __name__ == '__main__':
    main()            