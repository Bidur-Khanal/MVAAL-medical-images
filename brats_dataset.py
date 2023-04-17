import os
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy
import random
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode


class BraTSDataset(Dataset):

    def __init__(self, root_dir, split_dir="train/", flip=False, resize = None, scale= None, crop=None, version = 1, brightness = False, v_flip = False, rotation = False, random_crop = False, segmentation_type = "whole_tumor"):
        
        self.flip = flip
        self.scale = scale
        self.resize = resize
        self.crop = crop
        self.root_dir = root_dir
        self.rotation = rotation
        self.v_flip = v_flip
        self.brightness = brightness

        self.random_crop = random_crop
        self.split_dir = split_dir
        self.flair_dir = 'flair'
        self.t1_dir = 't1'
        self.mask_dir = 'mask'
        self.segmentation_type = segmentation_type
        self.version = version

        if version == 5:
            self.flair_files = numpy.load(self.root_dir+split_dir+'flair_seed0_HGG.npy', allow_pickle='TRUE')
            self.t1_files = numpy.load(self.root_dir+split_dir+'t1_seed0_HGG.npy', allow_pickle='TRUE')
            self.mask_files = numpy.load(self.root_dir+split_dir+'mask_seed0_HGG.npy', allow_pickle='TRUE')

        elif version == 6:
            self.flair_files = numpy.load(self.root_dir+split_dir+'flair_seed0_LGG.npy', allow_pickle='TRUE')
            self.t1_files = numpy.load(self.root_dir+split_dir+'t1_seed0_LGG.npy', allow_pickle='TRUE')
            self.mask_files = numpy.load(self.root_dir+split_dir+'mask_seed0_LGG.npy', allow_pickle='TRUE')

        elif version == 7:
            self.flair_files = numpy.load(self.root_dir+split_dir+'flair_seed0_HGG.npy', allow_pickle='TRUE')
            self.t1_files = numpy.load(self.root_dir+split_dir+'t1_seed0_HGG.npy', allow_pickle='TRUE')
            self.mask_files = numpy.load(self.root_dir+split_dir+'mask_seed0_HGG.npy', allow_pickle='TRUE')

            ### concat HGG and LGG
            self.flair_files = np.concatenate((self.flair_files,numpy.load(self.root_dir+split_dir+'flair_seed0_LGG.npy', allow_pickle='TRUE')))
            self.t1_files = np.concatenate((self.t1_files,numpy.load(self.root_dir+split_dir+'t1_seed0_LGG.npy', allow_pickle='TRUE')))
            self.mask_files = np.concatenate((self.mask_files,numpy.load(self.root_dir+split_dir+'mask_seed0_LGG.npy', allow_pickle='TRUE')))
            
    def __len__(self):
        return len(self.flair_files)
    
    def slice_normalize(self,slice):
        '''
            input: unnormalized slice 
            OUTPUT: normalized clipped slice
        '''

        # make sure that percentile below 1% and above 99% are cutoff
        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)

        return slice
    
    def __getitem__(self, idx):
        
        flair_image = self.slice_normalize(self.flair_files[idx])
        t1_image = self.slice_normalize(self.t1_files[idx])
        
        flair_image = (torch.tensor(flair_image, dtype = torch.float32)).unsqueeze(0)
        t1_image = (torch.tensor(t1_image, dtype = torch.float32)).unsqueeze(0)
        mask = (torch.tensor(self.mask_files[idx])).unsqueeze(0)
        

        if self.segmentation_type == "whole_tumor":
            ### make sure that we have only one class
            # replace 3 with 1
            mask[mask==4] = 1
            mask[mask==2] = 1

        if self.segmentation_type == "core_tumor":
            ### make sure that we have only one class
            mask[mask==4] = 1
            mask[mask==2] = 0

        if self.segmentation_type == "ET":
            ### make sure that we have only one class
            mask[mask==1] = 0
            mask[mask==2] = 0
            mask[mask==4] = 1
    
        flair_image, mask, t1_image = preprocess(flair_image, mask, t1_image, flip= self.flip, resize= self.resize, crop= self.crop, rotation = self.rotation, v_flip =self.v_flip, brightness = self.brightness, random_crop= self.random_crop)

        flair_image = (flair_image - flair_image.min())/(flair_image.max() - flair_image.min())
        t1_image =  (t1_image - t1_image.min())/(t1_image.max()-t1_image.min())


        if self.version == 5:
            # normalize the image to 0 mean and 1 std using HGG brats dataset statistics
            flair_image = (flair_image - 0.14407136)/0.26404217
            t1_image = (t1_image - 0.18798836)/0.3249902

        return  t1_image,mask.squeeze(0), flair_image
      

class BraTSDataset_3channel_input(Dataset):

    def __init__(self, root_dir, split_dir="train/", flip=False, resize = None, scale= None, crop=None, version = 1, brightness = False, v_flip = False, rotation = False, random_crop = False, segmentation_type = "whole_tumor"):
        
        self.flip = flip
        self.scale = scale
        self.resize = resize
        self.crop = crop
        self.root_dir = root_dir
        self.rotation = rotation
        self.v_flip = v_flip
        self.brightness = brightness

        self.random_crop = random_crop
        self.split_dir = split_dir
        self.flair_dir = 'flair'
        self.t1_dir = 't1'
        self.mask_dir = 'mask'
        self.segmentation_type = segmentation_type
        self.version = version

        if version == 5:
            self.flair_files = numpy.load(self.root_dir+split_dir+'flair_seed0_HGG.npy', allow_pickle='TRUE')
            self.t1_files = numpy.load(self.root_dir+split_dir+'t1_seed0_HGG.npy', allow_pickle='TRUE')
            self.mask_files = numpy.load(self.root_dir+split_dir+'mask_seed0_HGG.npy', allow_pickle='TRUE')

        elif version == 6:
            self.flair_files = numpy.load(self.root_dir+split_dir+'flair_seed0_LGG.npy', allow_pickle='TRUE')
            self.t1_files = numpy.load(self.root_dir+split_dir+'t1_seed0_LGG.npy', allow_pickle='TRUE')
            self.mask_files = numpy.load(self.root_dir+split_dir+'mask_seed0_LGG.npy', allow_pickle='TRUE')

        elif version == 7:
            self.flair_files = numpy.load(self.root_dir+split_dir+'flair_seed0_HGG.npy', allow_pickle='TRUE')
            self.t1_files = numpy.load(self.root_dir+split_dir+'t1_seed0_HGG.npy', allow_pickle='TRUE')
            self.mask_files = numpy.load(self.root_dir+split_dir+'mask_seed0_HGG.npy', allow_pickle='TRUE')

            ### concat HGG and LGG
            self.flair_files = np.concatenate((self.flair_files,numpy.load(self.root_dir+split_dir+'flair_seed0_LGG.npy', allow_pickle='TRUE')))
            self.t1_files = np.concatenate((self.t1_files,numpy.load(self.root_dir+split_dir+'t1_seed0_LGG.npy', allow_pickle='TRUE')))
            self.mask_files = np.concatenate((self.mask_files,numpy.load(self.root_dir+split_dir+'mask_seed0_LGG.npy', allow_pickle='TRUE')))
            
    def __len__(self):
        return len(self.flair_files)
    
    def slice_normalize(self,slice):
        '''
            input: unnormalized slice 
            OUTPUT: normalized clipped slice
        '''

        # make sure that percentile below 1% and above 99% are cutoff
        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)

        return slice
    
    def __getitem__(self, idx):

        flair_image = self.slice_normalize(self.flair_files[idx])
        t1_image = self.slice_normalize(self.t1_files[idx])
        
        flair_image = (torch.tensor(flair_image, dtype = torch.float32)).unsqueeze(0)
        t1_image = (torch.tensor(t1_image, dtype = torch.float32)).unsqueeze(0)
        mask = (torch.tensor(self.mask_files[idx])).unsqueeze(0)
        

        if self.segmentation_type == "whole_tumor":
            ### make sure that we have only one class
            # replace 3 with 1
            mask[mask==4] = 1
            mask[mask==2] = 1

        if self.segmentation_type == "core_tumor":
            ### make sure that we have only one class
            mask[mask==4] = 1
            mask[mask==2] = 0

        if self.segmentation_type == "ET":
            ### make sure that we have only one class
            mask[mask==1] = 0
            mask[mask==2] = 0
            mask[mask==4] = 1
    
        flair_image, mask, t1_image = preprocess(flair_image, mask, t1_image, flip= self.flip, resize= self.resize, crop= self.crop, rotation = self.rotation, v_flip =self.v_flip, brightness = self.brightness, random_crop= self.random_crop)

        flair_image = (flair_image - flair_image.min())/(flair_image.max() - flair_image.min())
        t1_image =  (t1_image - t1_image.min())/(t1_image.max()-t1_image.min())

        if self.version == 5:
                    # normalize the image to 0 mean and 1 std using HGG brats dataset statistics
                    flair_image = (flair_image - 0.14407136)/0.26404217
                    t1_image = (t1_image - 0.18798836)/0.3249902

        # broast cast all images to 3 channel
        flair_image = flair_image.repeat(3,1,1)
        t1_image = t1_image.repeat(3,1,1)

        return  t1_image,mask.squeeze(0), flair_image


def preprocess(flair, mask, t1,  flip=False, resize = None, crop=None, rotation = False, v_flip = False, brightness = False, random_crop = False):

    if crop:
        flair = F.center_crop(flair,crop)
        t1 = F.center_crop(t1,crop)
        mask = F.center_crop(mask,crop)

    if resize:
        previous = len(np.unique(mask))
        flair = F.resize(flair, size= resize, interpolation =InterpolationMode.BILINEAR, antialias = True)
        t1 = F.resize(t1, size= resize, interpolation =InterpolationMode.BILINEAR, antialias = True)
        mask = F.resize(mask, size= resize, interpolation =InterpolationMode.NEAREST, antialias = False)
        # if len(np.unique(mask)) != previous:
        #     print ("The classes has been changed due to downsampling")


    if random_crop:
        _,h,w = mask.shape
        
        if random.random() < 0.5:
            new_size = random.uniform(h-40,h-10)
            flair = F.center_crop(flair,new_size)
            t1 = F.center_crop(t1,new_size)
            mask = F.center_crop(mask,new_size)

        # make sure to resize back to the original size
        flair = F.resize(flair, size= (h,w), interpolation =InterpolationMode.BILINEAR, antialias = True)
        t1 = F.resize(t1, size= (h,w), interpolation =InterpolationMode.BILINEAR, antialias = True)
        mask = F.resize(mask, size= (h,w), interpolation =InterpolationMode.NEAREST, antialias = False)
        
    if flip:
        if random.random() < 0.5:
            flair = F.hflip(flair)
            t1 = F.hflip(t1)
            mask =F.hflip(mask)

    if v_flip:
        if random.random() < 0.5:
            flair = F.vflip(flair)
            t1 = F.vflip(t1)
            mask =F.vflip(mask)

    if rotation:
        if random.random() < 0.5:
            degree = random.uniform(-20,20)
            flair = F.rotate(flair, degree, interpolation = InterpolationMode.BILINEAR)
            t1 = F.rotate(t1, degree, interpolation = InterpolationMode.BILINEAR)
            mask = F.rotate(mask, degree, interpolation = InterpolationMode.NEAREST)

    # use these additional augmentations if necessary
    '''
    if brightness:
        brightness_factor = random.uniform(0.9,1.1)
        flair = F.adjust_brightness(flair, brightness_factor)
        t1 = F.adjust_brightness(t1, brightness_factor)

    if shear:
        if random.random() < 0.5:
            shear_degree = random.uniform(-10,10)
            flair = F.affine(flair, angle =0, translate = (0,0), interpolation = InterpolationMode.BILINEAR)
            t1 = F.affine(t1, degree, interpolation = InterpolationMode.BILINEAR)
            mask = F.affine(mask, degree, interpolation = InterpolationMode.NEAREST) 
    '''

    mask = mask.long()
    return flair, mask, t1
   

class BraTSDataset_classification(Dataset):

    def __init__(self, root_dir, split_dir = "train", version = 5, flip=False, resize = None, scale= None, crop=None, v_flip = False, rotation = False, random_crop = False):
        
        self.flip = flip
        self.scale = scale
        self.resize = resize
        self.crop = crop
        self.root_dir = root_dir
        self.rotation = rotation
        self.v_flip = v_flip
        self.random_crop = random_crop
        self.split_dir = split_dir
        self.version = version
     
        if version == 5:
            self.flair_files = numpy.load(self.root_dir+split_dir+'flair_seed0_HGG.npy', allow_pickle='TRUE')
            self.t1_files = numpy.load(self.root_dir+split_dir+'t1_seed0_HGG.npy', allow_pickle='TRUE')
            self.class_files = numpy.load(self.root_dir+split_dir+'multilabel_classes_seed0_HGG.npy', allow_pickle='TRUE')

        elif version == 6:
            self.flair_files = numpy.load(self.root_dir+split_dir+'flair_seed0_LGG.npy', allow_pickle='TRUE')
            self.t1_files = numpy.load(self.root_dir+split_dir+'t1_seed0_LGG.npy', allow_pickle='TRUE')
            self.class_files = numpy.load(self.root_dir+split_dir+'multilabel_classes_seed0_LGG.npy', allow_pickle='TRUE')


    def slice_normalize(self,slice):
        '''
            input: unnormalized slice 
            OUTPUT: normalized clipped slice
        '''

        # make sure that percentile below 1% and above 99% are cutoff
        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)

        return slice


    def __len__(self):
        return len(self.flair_files)
    
    def __getitem__(self, idx):

        flair_image = self.slice_normalize(self.flair_files[idx])
        t1_image = self.slice_normalize(self.t1_files[idx])
        
        flair_image = (torch.tensor(flair_image, dtype = torch.float32)).unsqueeze(0)
        t1_image = (torch.tensor(t1_image, dtype = torch.float32)).unsqueeze(0)
        classes = (torch.tensor(self.class_files[idx]))

        flair_image, t1_image = preprocess_classification(flair_image, t1_image, flip= self.flip, resize= self.resize, crop= self.crop,rotation = self.rotation, v_flip =self.v_flip, random_crop= self.random_crop)


        flair_image = (flair_image - flair_image.min())/(flair_image.max() - flair_image.min())
        t1_image =  (t1_image - t1_image.min())/(t1_image.max()-t1_image.min())


        # broast cast all images to 3 channel
        flair_image = flair_image.repeat(3,1,1)
        t1_image = t1_image.repeat(3,1,1)

        # ### normalize the image using the ImageNet Statistics
        MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])

        flair_image = (flair_image- MEAN[:, None, None]) / STD[:, None, None]
        t1_image = (t1_image - MEAN[:, None, None]) / STD[:, None, None]

        return flair_image, classes, t1_image
       
def preprocess_classification(flair, t1,  flip=False, resize = None, crop=None,rotation = False, v_flip = False, random_crop = False):

    if crop:
        flair = F.center_crop(flair,crop)
        t1 = F.center_crop(t1,crop)

    if resize:
        flair = F.resize(flair, size= resize, interpolation =InterpolationMode.BILINEAR, antialias = True)
        t1 = F.resize(t1, size= resize, interpolation =InterpolationMode.BILINEAR, antialias = True)

    if random_crop:
        _,h,w = flair.shape
        
        if random.random() < 0.5:
            new_size = random.uniform(h-40,h-10)
            flair = F.center_crop(flair,new_size)
            t1 = F.center_crop(t1,new_size)
        
        # make sure to resize back to the original size
        flair = F.resize(flair, size= (h,w), interpolation =InterpolationMode.BILINEAR, antialias = True)
        t1 = F.resize(t1, size= (h,w), interpolation =InterpolationMode.BILINEAR, antialias = True)
        
    if flip:
        if random.random() < 0.5:
            flair = F.hflip(flair)
            t1 = F.hflip(t1)

    if v_flip:
        if random.random() < 0.5:
            flair = F.vflip(flair)
            t1 = F.vflip(t1)
        
    if rotation:
        if random.random() < 0.5:
            degree = random.uniform(-20,20)
            flair = F.rotate(flair, degree, interpolation = InterpolationMode.BILINEAR)
            t1 = F.rotate(t1, degree, interpolation = InterpolationMode.BILINEAR)

    return flair, t1
   
   