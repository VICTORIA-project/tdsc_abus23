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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from importlib import import_module
import copy

from monai.transforms import (
    Compose,
    ScaleIntensityd,
    EnsureTyped,
    Resized,
)

# special imports
from datasets_utils.datasets import ABUS_test, slice_number
sys.path.append(str(repo_path / 'SAMed')) if str(repo_path / 'SAMed') not in sys.path else None
from SAMed.segment_anything import sam_model_registry

class USSegmentation:
    def __init__(self, checkpoint_path:Path or str):
        """initialization of the USSegmentation class

        Args:
            checkpoint_path (Pathorstr): path to the checkpoint file, for the SAM model
        """
        # HP
        num_classes = 1
        image_size = 512

        sam, _ = sam_model_registry['vit_b'](image_size=image_size,
                                                num_classes=num_classes,
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
                        Resized(keys=["image"], spatial_size=(image_size, image_size),mode=['area']),
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

    def process_image(self, input_image):
        image = self.test_transform({"image": input_image})["image"]
        image = image.to(device=self.device).unsqueeze(0)
        input_h_flipped = self.h_flip(image)
        final_output = torch.zeros((1, 4, 256, 256), dtype=torch.float32).to(self.device)
        for i in range(5):
            self.models[i].eval()
            outputs = self.models[i](image, True, 256)
            outputs_h_flip = self.models[i](input_h_flipped, True, 256)

            output_masks_t = (outputs['masks'] + self.h_flip(outputs_h_flip['masks'])) / 2
            final_output += output_masks_t

        output_masks = torch.argmax(torch.softmax(final_output / 5, dim=1), dim=1, keepdim=True)

        return output_masks