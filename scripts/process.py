# running as docker?
docker_running = True

# define repo path and add it to the path
from pathlib import Path
import sys, os
if not docker_running: # if we are running locally
    repo_path= Path.cwd().resolve()
    while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
        repo_path = repo_path.parent #go up one level
else: # if running in the container
    repo_path = Path('/opt/usuari')
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize, InterpolationMode
import shutil
from scipy.ndimage import label, generate_binary_structure


# special imports
from segmentation import USSegmentation

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

class lesion_seg:
    def __init__(self):
        # define paths
        self.input_dir = repo_path / 'input'# if docker_running else repo_path / 'input'
        print(f'The content of the input dir:{self.input_dir} is: {os.listdir(self.input_dir)}')
        self.output_dir = repo_path / 'predict' / 'Segmentation' #if docker_running else Path(repo_path / 'predict' / 'Segmentation')
        # show ls -la of repo_path
        self.output_dir.mkdir(parents=True, exist_ok=True) # make sure the output dir exists
        self.checkpoint_dir = repo_path / 'checkpoints' / 'sam_vit_b_01ec64.pth'
        self.cached_dir = repo_path / 'cached_data'
        self.cached_dir.mkdir(parents=True, exist_ok=True) # create cached dir in root
        self.slices_dir = self.cached_dir / 'slices'
        self.probs_dir = self.cached_dir / 'probs'
        self.seed_dir = self.cached_dir / 'seed'
        # load all folds models
        self.md = USSegmentation(self.checkpoint_dir)
        load_success = self.md.load_model()
        if load_success:
            print("Successfully loaded models")

    def save_slices(self, image_path:Path):
        """given an nrrd image path, the slices are saved in the cached_dir/slices folder

        Args:
            image_path (Path): Path to the nrrd image
        """
        # Expansion HP
        x_expansion = 865
        y_expansion = 865
        x_resizing = 512
        y_resizing = 512
        file_format = 'mha'


        # remove folder if exists, always starts from scratch
        self.slices_dir.mkdir(exist_ok=True, parents=True)

        # transforms
        preprocess_im = Compose(
                [
                    Resize((x_resizing, y_resizing), interpolation= InterpolationMode.BILINEAR),
                ]
        )

        # get image
        im_sitk = sitk.ReadImage(image_path)
        shape = im_sitk.GetSize()
        im = sitk.GetArrayFromImage(im_sitk)
        # now, we complete the images and labels to the expansion variables
        if im.shape[2]<x_expansion:
            # print('Expanding x dimension')
            im = np.concatenate((im, np.zeros((im.shape[0], im.shape[1], x_expansion-im.shape[2]), dtype=np.int8)), axis=2)

        if im.shape[1]<y_expansion:
            # print('Expanding y dimension')
            im = np.concatenate((im, np.zeros((im.shape[0], y_expansion-im.shape[1], im.shape[2]), dtype=np.int8)), axis=1)

        # all z values available
        z_values = np.array(range(im.shape[0]))
        for z in tqdm(z_values):
            # preprocess image
            im_slice = Image.fromarray(im[z])
            im_slice = preprocess_im(im_slice)
            im_slice = np.asarray(im_slice)
            # put channel first and repeat in RGB
            im_slice = np.repeat(np.expand_dims(im_slice, axis=0), 3, axis=0)

            # saving path
            save_name = f'slice_{z}.{file_format}'
            # save image
            sitk.WriteImage(sitk.GetImageFromArray(im_slice), str(self.slices_dir / save_name))
        
        return shape

    def prob_map(self, image_path:Path):
        """create a probability map for a given image path

        Args:
            image_path (Path): path of the nrrd original image
        """
        original_shape = self.save_slices(image_path) # save slices and get original shape
        prob_map = self.md.process_image(slices_dir=self.slices_dir, original_shape=original_shape)
        # save the prob map as numpy array
        self.probs_dir.mkdir(exist_ok=True, parents=True)
        np.save(self.probs_dir / 'prob_map.npy', prob_map)

    def seed_definition(self):
        """constructs and saves seed using the probability map already saved by prob_map method
        """

        # HP
        top_hat = 0.0001
        # create seed dir
        saving_dir_name = f'seed'
        saving_dir = self.cached_dir / saving_dir_name
        saving_dir.mkdir(parents=True, exist_ok=True)

        # load probs
        probs = np.load(self.probs_dir / 'prob_map.npy')

        # create seed
        seed = np.zeros_like(probs)

        # use top values as seed
        valid_pixels = probs>(np.max(probs)-top_hat)
        seed[valid_pixels] = 1
        seed = seed.astype(np.uint8)

        # get lcc
        seed = lcc(seed)

        # save as numpy
        saving_path = saving_dir / 'seed.npy'
        np.save(saving_path, seed)           

    def postprocess(self):
        # HP
        high_threshold = 0.65

        # load seed array
        seed = np.load(self.seed_dir / 'seed.npy')
        probs = np.load(self.probs_dir / 'prob_map.npy')

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

        return mask
    
    def segment(self):
        # given the images found in the input dir
        image_paths = list(self.input_dir.glob("*"))
        
        for image_path in image_paths:
            
            print(f'Processing patient: {image_path.name.split("_")[1].split(".")[0]}')
            if self.cached_dir.exists(): # always start from scratch
                shutil.rmtree(self.cached_dir)
            # create prob map
            self.prob_map(image_path)
            # create seed
            self.seed_definition()
            # postprocess
            mask = self.postprocess()
            # save
            mask = sitk.GetImageFromArray(mask)
            # write
            sitk.WriteImage(mask, str(self.output_dir / ('MASK_'+image_path.name.split('_')[1])))
        print(f'The content of the output dir:{self.output_dir} is: {os.listdir(self.output_dir)}')

if __name__ == "__main__":
    lesion_seg().segment()