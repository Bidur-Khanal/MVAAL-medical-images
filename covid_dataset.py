import os
import torch
from torch.utils.data import Dataset
import PIL.Image as Image
import random
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
import h5py


class COVID19_Xray_binary(Dataset):

    def __init__(self, root_dir, split_type="train", flip=False, resize = None, scale= None, crop=None, brightness = False, rotation = False, random_crop = False):
        
        self.flip = flip
        self.scale = scale
        self.resize = resize
        self.crop = crop
        self.root_dir = root_dir
        self.rotation = rotation
        self.brightness = brightness
        self.random_crop = random_crop

        self.split_type = split_type
       
        with h5py.File(os.path.join(self.root_dir, self.split_type+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]
            self.masks = hf["dataset"]["masks"][:]

    def __len__(self):
        return len(self.images)
    
    
    def __getitem__(self, idx):

        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
         
        index = int(idx)
        image, mask, target = self.images[index], self.masks[index],self.targets[index]
    

        image = (torch.tensor(image, dtype = torch.float32)).unsqueeze(0)
        mask = (torch.tensor(mask, dtype = torch.float32)).unsqueeze(0)
        target = torch.tensor(target).unsqueeze(0)
        
        image, mask = preprocess(image, mask, flip= self.flip, resize= self.resize, crop= self.crop, rotation = self.rotation, brightness = self.brightness, random_crop= self.random_crop)

        image = image/255.
        mask = mask/255.
        
        # normalize the image 
        image = (image-0.5)/0.5

        # broast cast all images to 3 channel
        image = image.repeat(3,1,1)
        
        return  image, mask, target
      


def preprocess(image, mask, flip=False, resize = None, crop=None, rotation = False, brightness = False, random_crop = False):

    if crop:
        image = F.center_crop(image,crop)
        mask = F.center_crop(mask,crop)
     
    if resize:
        image = F.resize(image, size= resize, interpolation =InterpolationMode.BILINEAR, antialias = True)
        mask = F.resize(mask, size= resize, interpolation =InterpolationMode.NEAREST, antialias = False)
     

    if random_crop:
        _,h,w = mask.shape
        
        if random.random() < 0.5:
            new_size = random.uniform(h-10,h-5)
            image = F.center_crop(image,new_size)
            mask = F.center_crop(mask,new_size)

        # make sure to resize back to the original size
        image = F.resize(image, size= (h,w), interpolation =InterpolationMode.BILINEAR, antialias = True)
        mask = F.resize(mask, size= (h,w), interpolation =InterpolationMode.NEAREST, antialias = False)
        
    if flip:
        if random.random() < 0.5:
            image = F.hflip(image)
            mask =F.hflip(mask)

    if rotation:
        if random.random() < 0.5:
            degree = random.uniform(-20,20)
            image = F.rotate(image, degree, interpolation = InterpolationMode.BILINEAR)
            mask = F.rotate(mask, degree, interpolation = InterpolationMode.NEAREST)

    return image, mask



class COVID19_Xray(Dataset):

    def __init__(self, root_dir, split_type="train", flip=False, resize = None, scale= None, crop=None, brightness = False, rotation = False, random_crop = False):
        
        self.flip = flip
        self.scale = scale
        self.resize = resize
        self.crop = crop
        self.root_dir = root_dir
        self.rotation = rotation
        self.brightness = brightness
        self.random_crop = random_crop

        self.split_type = split_type
       
        with h5py.File(os.path.join(self.root_dir, self.split_type+"_3class.hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]
            self.masks = hf["dataset"]["masks"][:]

    def __len__(self):
        return len(self.images)
    
    
    def __getitem__(self, idx):

        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
         
        index = int(idx)
        image, mask, target = self.images[index], self.masks[index],self.targets[index]
    

        image = (torch.tensor(image, dtype = torch.float32)).unsqueeze(0)
        mask = (torch.tensor(mask, dtype = torch.float32)).unsqueeze(0)
        target = torch.tensor(target)
        
        image, mask = preprocess(image, mask, flip= self.flip, resize= self.resize, crop= self.crop, rotation = self.rotation, brightness = self.brightness, random_crop= self.random_crop)

        image = image/255.
        mask = mask/255.
        
        # normalize the image 
        image = (image-0.5)/0.5

        # broast cast all images to 3 channel
        image = image.repeat(3,1,1)
        
        return  image, mask, target
      


def preprocess(image, mask, flip=False, resize = None, crop=None, rotation = False, brightness = False, random_crop = False):

    if crop:
        image = F.center_crop(image,crop)
        mask = F.center_crop(mask,crop)
     
    if resize:
        image = F.resize(image, size= resize, interpolation =InterpolationMode.BILINEAR, antialias = True)
        mask = F.resize(mask, size= resize, interpolation =InterpolationMode.NEAREST, antialias = False)
     

    if random_crop:
        _,h,w = mask.shape
        
        if random.random() < 0.5:
            new_size = random.uniform(h-10,h-5)
            image = F.center_crop(image,new_size)
            mask = F.center_crop(mask,new_size)

        # make sure to resize back to the original size
        image = F.resize(image, size= (h,w), interpolation =InterpolationMode.BILINEAR, antialias = True)
        mask = F.resize(mask, size= (h,w), interpolation =InterpolationMode.NEAREST, antialias = False)
        
    if flip:
        if random.random() < 0.5:
            image = F.hflip(image)
            mask =F.hflip(mask)

    if rotation:
        if random.random() < 0.5:
            degree = random.uniform(-20,20)
            image = F.rotate(image, degree, interpolation = InterpolationMode.BILINEAR)
            mask = F.rotate(mask, degree, interpolation = InterpolationMode.NEAREST)

    return image, mask
   