# running as docker?
docker_running = True

# define repo path and add it to the path
from pathlib import Path
import os, sys
if not docker_running: # if we are running locally
    repo_path= Path.cwd().resolve()
    while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
        repo_path = repo_path.parent #go up one level
else: # if running in the container
    repo_path = Path('/opt/usuari')
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import numpy as np
from importlib import import_module
import copy
from tqdm import tqdm
from PIL import Image
import torchvision
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    EnsureTyped,
    Resized,
)


# special imports
from datasets_utils.datasets import ABUS_test
sys.path.append(str(repo_path / 'SAMed')) if str(repo_path / 'SAMed') not in sys.path else None
from SAMed.segment_anything import sam_model_registry

class USSegmentation:
    def __init__(self, checkpoint_path:Path or str):
        """initialization of the USSegmentation class

        Args:
            checkpoint_path (Pathorstr): path to the checkpoint file, for the SAM model
        """
        # HP
        self.num_classes = 1
        self.image_size = 512

        sam, _ = sam_model_registry['vit_b'](image_size=self.image_size,
                                                num_classes=self.num_classes,
                                                checkpoint=str(checkpoint_path),
                                                pixel_mean=[0, 0, 0],
                                                pixel_std=[1, 1, 1])
        # load lora model
        pkg = import_module('sam_lora_image_encoder')
        model = pkg.LoRA_Sam(sam, 4)


        self.models = [copy.deepcopy(model) for i in range(5)]
        self.device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
        self.test_transform = Compose(
                    [
                        ScaleIntensityd(keys=["image"]),
                        Resized(keys=["image"], spatial_size=(self.image_size, self.image_size),mode=['area']),
                        EnsureTyped(keys=["image"])
                    ])

    def load_model(self):
        # optimum weight hard coded   
        optimum_weights = [
            'model_weights/fold0/epoch_19.pth',
            'model_weights/fold1/epoch_13.pth',
            'model_weights/fold2/epoch_5.pth',
            'model_weights/fold3/epoch_25.pth',
            'model_weights/fold4/epoch_14.pth',
        ]
        for i, model_path in enumerate(optimum_weights):
            # load weighs
            load_path = repo_path / model_path
            self.models[i].load_lora_parameters(str(load_path))
            self.models[i].eval()
            self.models[i].to(self.device)
        print("Models loaded on CUDA") if torch.cuda.is_available() else print("Model loaded on CPU")

        return True # for logical purposes

    def process_image(self, slices_dir, original_shape):

        # get all files in the folder in a list, only mha files
        slice_files = [file for file in os.listdir(slices_dir) if file.endswith('.mha')] # unordered files
        slice_files = sorted(slice_files, key=lambda x: int(x.split('.')[0].split('_')[1]))

        # create useful paths
        image_files = np.array([slices_dir / i for i in slice_files])
        db_val = ABUS_test(transform=self.test_transform,list_dir=image_files)   
        valloader = DataLoader(db_val, batch_size=32, shuffle=False, num_workers=12, pin_memory=True)

        # 2. Create probability volume
        accumulated_mask = torch.zeros((len(db_val),self.num_classes+1,self.image_size,self.image_size)) # store final mask per patient

        for model in self.models: # for each model learned

            model_mask = [] # for appending slices of same model
            for sample_batch in tqdm(valloader, total=len(valloader), desc='Processing slices'):
                with torch.no_grad():
                    # get data
                    image_batch = sample_batch["image"].to(self.device)
                    # forward and losses computing
                    outputs = model(image_batch, True, self.image_size)
                    # stack the masks
                    model_mask.append(outputs['masks'].detach().cpu())
            # stack tensors in a single one
            model_mask = torch.cat(model_mask, dim=0)
            accumulated_mask += model_mask
        print(f'The shape of the accumulated mask is {accumulated_mask.shape}')

        # get the mean
        accumulated_mask /= len(self.models)
        accumulated_mask = torch.softmax(accumulated_mask, dim=1)[:,1] # get lesion probability
        accumulated_mask = accumulated_mask.cpu().numpy()

        # reshape each slice
        x_expansion = 865
        y_expansion = 865
        resized_mask = []
        for slice_num in tqdm(range(accumulated_mask.shape[0])):
            im_slice = accumulated_mask[slice_num,:,:]
            im_slice = Image.fromarray(im_slice)
            im_slice_comeback = torchvision.transforms.Resize(
                (x_expansion, y_expansion),
                interpolation= torchvision.transforms.InterpolationMode.BILINEAR, # bilineal or nearest? probs bilineal
                )(im_slice)
            resized_mask.append(im_slice_comeback)
        # stack all slices
        resized_mask = np.stack(resized_mask, axis=0)
        # get original size and save
        final_mask = resized_mask[:,:original_shape[1],:original_shape[0]]
        print(f'The shape of the final output is {final_mask.shape}')

        return final_mask