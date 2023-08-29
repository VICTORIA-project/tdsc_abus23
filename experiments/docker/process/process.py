# running as docker?
docker_running = False

# define repo path and add it to the path
from pathlib import Path
import sys, os
if not docker_running: # if we are running locally
    repo_path= Path.cwd().resolve()
    while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
        repo_path = repo_path.parent #go up one level
else: # if running in the container
    repo_path = Path('opt/usuari')
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None
import SimpleITK
import numpy as np
import torch

# special imports
from segmentation import USSegmentation


class lesion_seg:
    def __init__(self):

        self.input_dir = Path('./input/') if docker_running else repo_path / 'input'
        self.output_dir = Path('./predict') / 'Segmentation' if docker_running else Path(repo_path / 'predict' / 'Segmentation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = repo_path / 'checkpoints' / 'sam_vit_b_01ec64.pth'
        self.md = USSegmentation(self.checkpoint_dir)
        load_success = self.md.load_model()
        if load_success:
            print("Successfully loaded models")

    def load_sitk(self, image_path) -> SimpleITK.Image:
        image = SimpleITK.ReadImage(str(image_path))
        return image

    def predict(self, input_image: SimpleITK.Image):
        # Obtain input image
        image_data = SimpleITK.GetArrayFromImage(input_image)
        with torch.no_grad():
            # Put it into the network for processing
            pred = self.md.process_image(image_data)
            # Post processing and saving of predicted images
            pred = pred.squeeze()
            pred = pred.cpu().numpy().astype(np.uint8)
            pred = SimpleITK.GetImageFromArray(pred)

            return pred

    def write_outputs(self, image_name, outputs):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        SimpleITK.WriteImage(outputs, os.path.join(self.output_dir, image_name + '.mha'))

    def process(self):
        image_paths = list(self.input_dir.glob("*"))
        for image_path in image_paths:
            image_name = os.path.basename(image_path).split('.')[0]
            image_sitk = self.load_sitk(image_path)
            result = self.predict(image_sitk)
            self.write_outputs(image_name, result)
