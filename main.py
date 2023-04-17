import numpy as np
import torch
import torchvision.models as torch_models
import torch.utils.data as data
import wandb
import torch.backends.cudnn as cudnn
from covid_dataset import COVID19_Xray_binary, COVID19_Xray
from brats_dataset import BraTSDataset, BraTSDataset_classification, BraTSDataset_3channel_input
import random
import os
import model
import multi_modal_model
from VAAL_solver import VAAL_Solver
from multimodal_VAAL_solver import multi_modal_VAAL_Solver
import arguments
from unet import UNet
from task_solver import train_task
from multi_label_classification_task_solver import train_multilabel_classifier
from binary_classification import train_classifier
from multiclass_classification import train_multiclass_classifier
from freeze_layers import LeNet_Freeze_Upto, ResNet18_Freeze_Upto
from sampler import EntropySampler

## Set Seed
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    cudnn.deterministic = True

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

def main(args):

    # (Initialize logging)
    
    experiment = wandb.init(project='MVAAL_trials')

    if args.dataset == 'brats_MIUA_HGG':
        
        crop = tuple(int(i) for i in args.crop.split(","))
        if min(crop) == 0:
            crop = None
        train_dataset =  BraTSDataset("data/BraTS_frames/MIUA/", flip = True, split_dir ="train/", resize= args.resize, crop = crop, version= 5,v_flip = True, brightness = True, rotation = True, random_crop= True, segmentation_type = args.segmentation_type)
        query_train_dataset =  BraTSDataset("data/BraTS_frames/MIUA/", flip = False, split_dir ="train/", resize= 128, crop = (210,210), version= 5,v_flip = False, brightness = False, rotation = False, random_crop= False,segmentation_type = args.segmentation_type)
        test_dataset = BraTSDataset("data/BraTS_frames/MIUA/", flip = False, split_dir = "test/", resize= args.resize, crop = crop, version= 5,segmentation_type = args.segmentation_type)
        val_dataset = BraTSDataset("data/BraTS_frames/MIUA/", flip = False, split_dir = "val/", resize= args.resize, crop = crop, version= 5,segmentation_type = args.segmentation_type)
        
        args.num_val = len(val_dataset)
        args.num_images = len(train_dataset)
        args.budget = 100
        args.initial_budget = 200
        args.num_classes = 2
        args.num_channels= 1

    elif args.dataset == 'brats_MIUA_HGG_3channel':
        
        crop = tuple(int(i) for i in args.crop.split(","))
        if min(crop) == 0:
            crop = None
        train_dataset =  BraTSDataset("data/BraTS_frames/MIUA/", flip = True, split_dir ="train/", resize= args.resize, crop = crop, version= 5,v_flip = True, brightness = True, rotation = True, random_crop= True, segmentation_type = args.segmentation_type)
        query_train_dataset =  BraTSDataset_3channel_input("data/BraTS_frames/MIUA/", flip = False, split_dir ="train/", resize= 128, crop = (210,210), version= 5,v_flip = False, brightness = False, rotation = False, random_crop= False,segmentation_type = args.segmentation_type)
        test_dataset = BraTSDataset("data/BraTS_frames/MIUA/", flip = False, split_dir = "test/", resize= args.resize, crop = crop, version= 5,segmentation_type = args.segmentation_type)
        val_dataset = BraTSDataset("data/BraTS_frames/MIUA/", flip = False, split_dir = "val/", resize= args.resize, crop = crop, version= 5,segmentation_type = args.segmentation_type)
        
        args.num_val = len(val_dataset)
        args.num_images = len(train_dataset)
        args.budget = 100
        args.initial_budget = 200
        args.num_classes = 2
        args.num_channels= 1
        args.query_channels = 3

    elif args.dataset == 'brats_MIUA_HGG_3channel_classification':
    
        crop = tuple(int(i) for i in args.crop.split(","))
        if min(crop) == 0:
            crop = None
        train_dataset =  BraTSDataset_classification("data/BraTS_frames/MIUA/", flip = True, split_dir ="train/", resize= args.resize, crop = crop, version= 5,v_flip = True, rotation = True, random_crop= True)
        query_train_dataset =  BraTSDataset_classification("data/BraTS_frames/MIUA/", flip = False, split_dir ="train/", resize= 128, crop = (210,210), version= 5,v_flip = False, rotation = False, random_crop= False)
        test_dataset = BraTSDataset_classification("data/BraTS_frames/MIUA/", flip = False, split_dir = "test/", resize= args.resize, crop = crop, version= 5)
        val_dataset = BraTSDataset_classification("data/BraTS_frames/MIUA/", flip = False, split_dir = "val/", resize= args.resize, crop = crop, version= 5)
        
        args.num_val = len(val_dataset)
        args.num_images = len(train_dataset)
        args.budget = 100
        args.initial_budget = 200
        args.num_classes = 3
        args.num_channels= 1
        args.query_channels = 3

    elif args.dataset == 'brats_MIUA_LGG':
        
        crop = tuple(int(i) for i in args.crop.split(","))
        if min(crop) == 0:
            crop = None
        train_dataset =  BraTSDataset("data/BraTS_frames/MIUA/", flip = True, split_dir ="train/", resize= args.resize, crop = crop, version= 6,v_flip = True, brightness = True, rotation = True, random_crop= True,segmentation_type = args.segmentation_type)
        query_train_dataset =  BraTSDataset("data/BraTS_frames/MIUA/", flip = False, split_dir ="train/", resize= 128, crop = (210,210), version= 6,v_flip = False, brightness = False, rotation = False, random_crop= False,segmentation_type = args.segmentation_type)
        test_dataset = BraTSDataset("data/BraTS_frames/MIUA/", flip = False, split_dir = "test/", resize= args.resize, crop = crop, version= 6,segmentation_type = args.segmentation_type)
        val_dataset = BraTSDataset("data/BraTS_frames/MIUA/", flip = False, split_dir = "val/", resize= args.resize, crop = crop, version= 6,segmentation_type = args.segmentation_type)
        
        args.num_val = len(val_dataset)
        args.num_images = len(train_dataset)
        args.budget = 100
        args.initial_budget = 200
        args.num_classes = 2
        args.num_channels= 1

    elif args.dataset == 'brats_MIUA_HGG_and_LGG':
        
        crop = tuple(int(i) for i in args.crop.split(","))
        if min(crop) == 0:
            crop = None
        train_dataset =  BraTSDataset("data/BraTS_frames/MIUA/", flip = True, split_dir ="train/", resize= args.resize, crop = crop, version= 7,v_flip = True, brightness = True, rotation = True, random_crop= True,segmentation_type = args.segmentation_type)
        query_train_dataset =  BraTSDataset("data/BraTS_frames/MIUA/", flip = False, split_dir ="train/", resize= 128, crop = (210,210), version= 7,v_flip = False, brightness = False, rotation = False, random_crop= False,segmentation_type = args.segmentation_type)
        test_dataset = BraTSDataset("data/BraTS_frames/MIUA/", flip = False, split_dir = "test/", resize= args.resize, crop = crop, version= 7,segmentation_type = args.segmentation_type)
        val_dataset = BraTSDataset("data/BraTS_frames/MIUA/", flip = False, split_dir = "val/", resize= args.resize, crop = crop, version= 7,segmentation_type = args.segmentation_type)
        
        args.num_val = len(val_dataset)
        args.num_images = len(train_dataset)
        args.budget = 300
        args.initial_budget = 200
        args.num_classes = 2
        args.num_channels= 1


    elif args.dataset == 'COVID_dataset_binary':
        
        crop = tuple(int(i) for i in args.crop.split(","))
        if min(crop) == 0:
            crop = None
        train_dataset =  COVID19_Xray_binary("data/covid_xray/", split_type ="train", resize= args.resize, crop = crop, flip = True, brightness = True, rotation = False, random_crop= True)
        query_train_dataset =  COVID19_Xray_binary("data/covid_xray/", split_type ="train", resize= 128, crop = crop, flip = True, brightness = True, rotation = False, random_crop= True)
        test_dataset = COVID19_Xray_binary("data/covid_xray/", split_type ="test", resize= args.resize, crop = crop, flip = False, brightness = False, rotation = False, random_crop= False)
        val_dataset = COVID19_Xray_binary("data/covid_xray/", split_type ="val", resize= args.resize, crop = crop, flip = False, brightness = False, rotation = False, random_crop= False)
        
        args.num_val = len(val_dataset)
        args.num_images = len(train_dataset)
        args.budget = 100
        args.initial_budget = 100
        args.num_classes = 1
        args.num_channels= 3
        args.query_channels = 3

    elif args.dataset == 'COVID_dataset_3_classes':

        crop = tuple(int(i) for i in args.crop.split(","))
        if min(crop) == 0:
            crop = None
        train_dataset =  COVID19_Xray("data/covid_xray/", split_type ="train", resize= args.resize, crop = crop, flip = True, brightness = True, rotation = False, random_crop= True)
        query_train_dataset =  COVID19_Xray("data/covid_xray/", split_type ="train", resize= 128, crop = crop, flip = True, brightness = True, rotation = False, random_crop= True)
        test_dataset = COVID19_Xray("data/covid_xray/", split_type ="test", resize= args.resize, crop = crop, flip = False, brightness = False, rotation = False, random_crop= False)
        val_dataset = COVID19_Xray("data/covid_xray/", split_type ="val", resize= args.resize, crop = crop, flip = False, brightness = False, rotation = False, random_crop= False)
        
        args.num_val = len(val_dataset)
        args.num_images = len(train_dataset)
        args.budget = 100
        args.initial_budget = 100
        args.num_classes = 3
        args.num_channels= 3
        args.query_channels = 3

    else:
        raise NotImplementedError

    #### if the num_classes is 2, we use binary cross entorpy, and the number of channel in segmentation will be 1.
    if args.task_type == "segmentation":
        if args.num_classes == 2:
            args.num_classes = 1

    # save the hyper-parameters in wandb
    experiment.config.update(vars(args))
    
    all_indices = np.arange(args.num_images)
    
    if args.train_full:
        initial_indices = all_indices.tolist()
    else:
        initial_indices = random.sample(list(all_indices), args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    

    querry_dataloader = FastDataLoader(query_train_dataset, sampler=sampler, 
            batch_size=args.batch_size, drop_last=True,num_workers=2, persistent_workers= True)
    train_dataloader = FastDataLoader(train_dataset, sampler=sampler, 
            batch_size=args.task_batch_size, drop_last=False,num_workers=2, persistent_workers= True)
    val_dataloader = FastDataLoader(val_dataset,
            batch_size=args.task_batch_size, drop_last=False,num_workers=2, persistent_workers= True)
    test_dataloader = FastDataLoader(test_dataset,
            batch_size=args.task_batch_size, drop_last=False,num_workers=2, persistent_workers= True)
       
    args.device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu')

    splits = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
   
    current_indices = list(initial_indices)
    for i, split in enumerate(splits):


        if args.with_replacement:
            args.budget = args.budget * (i+1) 


        experiment.log({
                    'split': split,
        
                })

        fix_seed(args.seed)

        if args.task_type == "segmentation":
            task_model = UNet(n_channels=args.num_channels, n_classes=args.num_classes)
            task_model.load_state_dict(torch.load('unet_init.pth'))
            task_model.to(device=args.device)
            train_task(args, net=task_model, train_loader = train_dataloader, val_loader = val_dataloader, test_loader= test_dataloader,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    amp=args.amp, wandb_log= experiment, split = split)

        elif args.task_type == "multi_label_classification":
           
            # unfreeze_layers = ["layer4.1","fc.weight", "fc.bias"]
            # task_model = ResNet18_Freeze_Upto(pretrained= True,unfreeze_layer= unfreeze_layers, num_classes= args.num_classes)

            task_model = torch_models.resnet18(pretrained= True)
            task_model.fc = torch.nn.Linear(512, args.num_classes)

            task_model.to(device=args.device)
            train_multilabel_classifier(args, net=task_model, train_loader = train_dataloader, val_loader = val_dataloader, test_loader= test_dataloader,
              epochs = args.epochs,
              batch_size = args.batch_size,
              learning_rate=args.lr,
              wandb_log= experiment, split = split)
            
        elif args.task_type == "multi_class_classification":

            # unfreeze_layers = ["layer4.1","fc.weight", "fc.bias"]
            # task_model = ResNet18_Freeze_Upto(pretrained= True,unfreeze_layer= unfreeze_layers, num_classes= args.num_classes)
            task_model = torch_models.resnet18(pretrained= True)
            task_model.fc = torch.nn.Linear(512, args.num_classes)

            task_model.to(device=args.device)
            train_multiclass_classifier(args, net=task_model, train_loader = train_dataloader, val_loader = val_dataloader, test_loader= test_dataloader,
                epochs = args.epochs,
                batch_size = args.batch_size,
                learning_rate=args.lr,
                wandb_log= experiment, split = split)
            
        elif args.task_type == "binary_classification":

            # unfreeze_layers = ["layer4.1","fc.weight", "fc.bias"]
            # task_model = ResNet18_Freeze_Upto(pretrained= True,unfreeze_layer= unfreeze_layers, num_classes= args.num_classes)
            task_model = torch_models.resnet18(pretrained= True)
            task_model.fc = torch.nn.Linear(512, args.num_classes)

            task_model.to(device=args.device)
            train_classifier(args, net=task_model, train_loader = train_dataloader, val_loader = val_dataloader, test_loader= test_dataloader,
                epochs = args.epochs,
                batch_size = args.batch_size,
                learning_rate=args.lr,
                wandb_log= experiment, split = split)
        

        if args.train_full:
            break

        ## all unlabeled train samples
        if args.with_replacement:
            unlabeled_indices = np.setdiff1d(list(all_indices), initial_indices)
        else:
            unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(query_train_dataset, 
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=True)

        
        if split == splits[-1]:
            break

        # if there is already saved indices for this experiment, then directly load that
        path = 'checkpoints/'+args.expt
        if os.path.exists(f'{path}/sampled_indices_after{split}.npy'):
            sampled_indices = np.load(f'{path}/sampled_indices_after{split}.npy')

        else:

            if args.method == 'VAAL':
                #### initilaize the VAAL models
                VAAL_solver = VAAL_Solver(args, test_dataloader)
                vae = model.VAE(args.latent_dim,nc=args.query_channels)


                # VAAL_solver = VAAL_Solver_encoder(args, test_dataloader)
                # vae = model_autoencoder.VAE(args.latent_dim,nc=args.num_channels)

                discriminator = model.Discriminator(args.latent_dim)

                vae = vae.to(device = args.device)
                discriminator = discriminator.to(device = args.device)

                #train the models on the current data
                vae, discriminator = VAAL_solver.train(split,querry_dataloader,
                                                    val_dataloader,
                                                    vae, 
                                                    discriminator,
                                                    unlabeled_dataloader,wandb_log=experiment)


                sampled_indices = VAAL_solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader, unlabeled_indices)


            elif args.method == 'multimodal_VAAL':
                
                #### initilaize the VAAL models
                multimodal_VAAL_solver = multi_modal_VAAL_Solver(args,test_dataloader)
                vae = multi_modal_model.VAE(args.latent_dim, nc=args.query_channels)
                discriminator = multi_modal_model.Discriminator(args.latent_dim)

                vae = vae.to(device = args.device)
                discriminator = discriminator.to(device = args.device)

                # train the models on the current data
                vae, discriminator = multimodal_VAAL_solver.train(split,querry_dataloader,
                                                    val_dataloader,
                                                    vae, 
                                                    discriminator,
                                                    unlabeled_dataloader)
                sampled_indices = multimodal_VAAL_solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader, unlabeled_indices)


            elif args.method == "RandomSampling":
                
                random.seed(args.random_sampling_seed)
                random.shuffle(unlabeled_indices)
                sampled_indices = unlabeled_indices[:args.budget]
                
        
            elif args.method == "EntropySampling":

                '''
                TODO: not implemented
                
                entropy_sampler = EntropySampler(args.budget)
                sampled_indices = entropy_sampler.sample(task_model,unlabeled_dataloader, unlabeled_indices, args.device)

                '''

            # save the current indices 
            path = 'checkpoints/'+args.expt
            np.save(f'{path}/sampled_indices_after{split}.npy', sampled_indices)
                
        if args.with_replacement:
            current_indices = list(initial_indices) + list(sampled_indices)
        else:
            current_indices = list(current_indices) + list(sampled_indices)

        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(query_train_dataset, sampler=sampler, 
                batch_size=args.batch_size, drop_last=True)

        train_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
                batch_size=args.batch_size, drop_last=False)

if __name__ == '__main__':
    args = arguments.get_args()
    fix_seed(args.seed)
    main(args)

