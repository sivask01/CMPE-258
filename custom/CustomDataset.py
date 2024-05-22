import os
import torch.utils.data
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
from torch import nn
import torchvision

class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir,transform=None):
        self.root = root_dir
        self.thermal_imgs = list(sorted(os.listdir(os.path.join(root_dir,'thermal_train/'))))
        self.rgb_imgs = list(sorted(os.listdir(os.path.join(root_dir,'rgb_train/'))))
        self.transforms = transform
        
    def __len__(self):
        return len(self.thermal_imgs)

    def __getitem__(self, idx):
        thermal_img_path = os.path.join(self.root,'thermal_train/', self.thermal_imgs[idx])
        thermal_img = Image.open(thermal_img_path)
        
        if self.transforms is not None:
            thermal_img = self.transforms(thermal_img)
        else:
            thermal_img = np.array(thermal_img)
            thermal_img = np.stack((thermal_img,)*3, axis=-1)
            thermal_img = torchvision.transforms.ToTensor()(thermal_img)


        rgb_img_path = os.path.join(self.root,'rgb_train/', self.rgb_imgs[idx])
        rgb_img = Image.open(rgb_img_path)
        
        if self.transforms is not None:
            rgb_img = self.transforms(rgb_img)
        else:
            rgb_img = np.array(rgb_img)
            rgb_img = torchvision.transforms.ToTensor()(rgb_img)

        
        return rgb_img,thermal_img