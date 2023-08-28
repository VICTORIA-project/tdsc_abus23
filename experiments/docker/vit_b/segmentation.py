# running as docker?
docker_running = False

# define repo path and add it to the path
from pathlib import Path
import os, sys
if not docker_running: # if we are running locally
    repo_path = Path('/home/ricardo/ABUS2023_documents/tdsc_abus23')
else: # if running in the container
    repo_path = Path('opt/usuari')
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

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
import SimpleITK as sitk
from PIL import Image
import torchvision
import pandas as pd

# special imports
from datasets_utils.datasets import ABUS_test, slice_number
sys.path.append(str(repo_path / 'SAMed')) if str(repo_path / 'SAMed') not in sys.path else None
from SAMed.segment_anything import sam_model_registry

def main():
    # HP
    batch_size = 16
    num_classes = 1
    image_size = 512

    # get SAM model
    checkpoint_dir = repo_path / 'checkpoints'
    sam, _ = sam_model_registry['vit_b'](image_size=image_size,
                                        num_classes=num_classes,
                                        checkpoint=str(checkpoint_dir / 'sam_vit_b_01ec64.pth'),
                                        pixel_mean=[0, 0, 0],
                                        pixel_std=[1, 1, 1])
    # load lora model
    pkg = import_module('sam_lora_image_encoder')
    model = pkg.LoRA_Sam(sam, 4)

    # define optimum weights
    optimum_weights = [
        'model_weights/fold0/epoch_19.pth',
        'model_weights/fold1/epoch_13.pth',
        'model_weights/fold2/epoch_5.pth',
        'model_weights/fold3/epoch_25.pth',
        'model_weights/fold4/epoch_14.pth',
    ]

    # transformations used on th images
    test_transform = Compose(
                [
                    ScaleIntensityd(keys=["image"]),
                    Resized(keys=["image"], spatial_size=(image_size, image_size),mode=['area']),
                    EnsureTyped(keys=["image"])
                ])
    
    metadata_path = repo_path / 'data/challange_2023/Test/metadata.csv'
    metadata = pd.read_csv(metadata_path)


    for pat_id in range(100,130,1): # each val id
        patient_meta = metadata[metadata['case_id'] == pat_id]
        original_shape = patient_meta['shape'].apply(lambda x: tuple(map(int, x[1:-1].split(',')))).values[0]

        # get data
        root_path = repo_path / 'data/challange_2023/Val/full-slice_512x512_all'
        path_images = (root_path / "image_mha")
        # get all files in the folder in a list, only mha files
        image_files = [file for file in os.listdir(path_images) if file.endswith('.mha')] # unordered files
        # # now, we will check if the path has at least one of the ids in the train_ids list
        val_files = [file for file in image_files if f'id_{pat_id}_' in file]
        val_files = sorted(val_files, key=slice_number) # sort them
        # # create final paths
        image_files = np.array([path_images / i for i in val_files])
        db_val = ABUS_test(transform=test_transform,list_dir=image_files)   
        valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        print(f'The patient id is {pat_id}')
        print(f'The number of slices is {len(db_val)}')
        # store final mask per patient
        accumulated_mask = torch.zeros((len(db_val),num_classes+1,image_size,image_size))
        
        for model_path in optimum_weights: # for each model learned
        
            # load weighs
            load_path = repo_path / model_path
            model.load_lora_parameters(str(load_path))
            model.eval()
            model.to(device);

            model_mask = []
            for sample_batch in valloader: # get some slides
                with torch.no_grad():
                    # get data
                    image_batch = sample_batch["image"].to(device)
                    # forward and losses computing
                    outputs = model(image_batch, True, image_size)
                    # stack the masks
                    model_mask.append(outputs['masks'].detach().cpu())
            # stack tensors in a single one
            model_mask = torch.cat(model_mask, dim=0)
            print(f'The shape of the independent model mask is {model_mask.shape}')
            accumulated_mask += model_mask

        # get the mean
        accumulated_mask /= len(optimum_weights)
        accumulated_mask = torch.softmax(accumulated_mask, dim=1)[:,1] # get lesion probability
        accumulated_mask = accumulated_mask.cpu().numpy()

        # reshape each slice
        x_expansion = 865
        y_expansion = 865
        resized_mask = []
        for slice_num in range(accumulated_mask.shape[0]):
            im_slice = accumulated_mask[slice_num,:,:]
            im_slice = Image.fromarray(im_slice)
            im_slice_comeback = torchvision.transforms.Resize(
                (x_expansion, y_expansion),
                interpolation= torchvision.transforms.InterpolationMode.BILINEAR, # bilineal or nearest? probs bilineal
                )(im_slice)
            resized_mask.append(im_slice_comeback)
        # stack all slices
        resized_mask = np.stack(resized_mask, axis=0)
        print(f'The shape of the resized mask is {resized_mask.shape}')

        # get original size
        final_mask = resized_mask[:,:original_shape[1],:original_shape[0]]
        print(f'The shape of the final output is {final_mask.shape}')

        saving_dir = repo_path / 'experiments/inference/segmentation/data/predictions' / 'full-size' / 'multi-model_probs'
        saving_dir.mkdir(parents=True, exist_ok=True)
        saving_path = saving_dir  / f'MASK_{pat_id}.nii.gz'

        # save the mask as nii.gz
        sitk.WriteImage(sitk.GetImageFromArray(final_mask), str(saving_path))

if __name__ == '__main__':
    main()