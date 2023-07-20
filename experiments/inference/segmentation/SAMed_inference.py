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
from tqdm import tqdm
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
    load_path = repo_path / 'experiments/SAMed_ABUS/results/vanilla_3class/fold0/weights/epoch_73.pth'
    model.load_lora_parameters(str(load_path))
    model.eval()
    model.to(device)

    # create fold
    kf = KFold(n_splits=5,shuffle=True,random_state=0)
    for fold_n, (_, val_ids) in enumerate(kf.split(range(100))):
        break
    print(f'The number of patients in the validation set is {len(val_ids)}')
    print(f'The patient ids in the validation set are {val_ids}')
    # transform
    val_transform = Compose(
            [
                EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'),

                ScaleIntensityd(keys=["image"]),

                Resized(keys=["image", "label"], spatial_size=(256, 256),mode=['area','nearest']),
                EnsureTyped(keys=["image"])
            ])

    # HP
    batch_size = 16

    patients_jaccard = np.zeros((len(val_ids), 2))
    patients_dice = np.zeros((len(val_ids), 2))
    for pat_num in range(len(val_ids)):
        pat_id = [val_ids[pat_num]]

        # get data
        root_path = repo_path / 'data/challange_2023/with_lesion'
        path_images = (root_path / "image_mha")
        path_labels = (root_path / "label_mha")
        # get all files in the folder in a list, only mha files
        image_files = sorted([file for file in os.listdir(path_images) if file.endswith('.mha')])
        # now, we will check if the path has at least one of the ids in the train_ids list
        val_files = [file for file in image_files if any(f'id_{id}_' in file for id in pat_id)]
        # create final paths
        image_files = np.array([path_images / i for i in val_files])
        label_files = np.array([path_labels / i for i in val_files])
        list_val = [image_files, label_files] # this is what we will pass to the dataset <-

        # define dataset and dataloader
        db_val = ABUS_dataset(transform=val_transform,list_dir=list_val)   
        valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        print(f'The patient id is {pat_id[0]}')
        print(f'The number of slices is {len(db_val)}')
        labels_array = []
        preds_array = []
        for sample_batch in valloader:
            # get data
            image_batch, label_batch = sample_batch["image"].to(device), sample_batch["label"].to(device)
            # forward and losses computing
            outputs = model(image_batch, True, 256)
            output_masks = outputs['masks'].detach().cpu()
            output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=False)

            #label_batch and output_masks in array
            image_batch = image_batch[:,0].cpu().numpy()
            label_batch = label_batch.cpu().numpy()
            output_masks = output_masks.cpu().numpy()
            # append to list
            labels_array.append(label_batch)
            preds_array.append(output_masks)
            
        # get 3D jaccard score
        labels_array = np.concatenate(labels_array)
        preds_array = np.concatenate(preds_array)
        jaccard_value = jaccard_score(labels_array.flatten(), preds_array.flatten())
        # dice from jaccard
        dice_value = 2*jaccard_value/(1+jaccard_value)
        print(f'Jaccard score is {jaccard_value}')
        print(f'Dice score is {dice_value}')
        print('-------------------------')
        # store in array
        patients_jaccard[pat_num, 0] = pat_id[0]
        patients_jaccard[pat_num, 1] = jaccard_value
        patients_dice[pat_num, 0] = pat_id[0]
        patients_dice[pat_num, 1] = dice_value
    # compute mean dice
    mean_dice = np.mean(patients_dice[:, 1])
    print(f'The mean dice is {mean_dice}')
if __name__ == "__main__":
    main()


        