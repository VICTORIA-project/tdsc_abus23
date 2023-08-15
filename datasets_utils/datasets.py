from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
from monai.transforms import (
    Compose,
    Resized,
)

class ABUS_dataset(Dataset):
    
    def __init__(self, list_dir:list, transform=None, spatial_size=(64, 64)):
        """from a list of directories and a transform, create the ABUS dataset

        Args:
            list_dir (list): list of two numpy arrays, each for the images and the labels
            transform (transforms, optional): MONAI of torch transforms. Defaults to None.
            spatial_size (tuple, optional): size of the low resolution labels. Defaults to (64, 64).
        """
        self.transform = transform  # using transform in torch!
        images = [sitk.GetArrayFromImage(sitk.ReadImage(str(i))) for i in list_dir[0]]
        labels = [sitk.GetArrayFromImage(sitk.ReadImage(str(i))) for i in list_dir[1]]

        self.sample_list = np.array(list(zip(images,labels)), dtype=object) # more efficient?
        
        self.resize=Compose([Resized(keys=["label"], spatial_size=spatial_size,mode=['nearest'])])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        
        if self.transform:
            sample=self.transform({"image": self.sample_list[idx][0], "label": self.sample_list[idx][1]})
        
        sample['low_res_label']=self.resize({"label":sample['label']})['label'][0]
        sample['label']=sample['label'][0]
        return sample
    
class ABUS_test(Dataset):
    
    def __init__(self, list_dir:list, transform=None):
        """from a list of directories and a transform, create the ABUS dataset

        Args:
            list_dir (list): list of two numpy arrays, each for the images and the labels
            transform (transforms, optional): MONAI of torch transforms. Defaults to None.
        """
        self.transform = transform  # using transform in torch!
        images = [sitk.GetArrayFromImage(sitk.ReadImage(str(i))) for i in list_dir]

        self.sample_list = np.array(images)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.transform:
            sample=self.transform({"image": self.sample_list[idx]})
        return sample
    
