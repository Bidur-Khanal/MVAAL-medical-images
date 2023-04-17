import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.utils.data as data
import sampler as query_Sampler
import numpy as np
import random
import model
import multi_modal_model
import arguments
from sklearn.manifold import TSNE
import seaborn as sns
import torch.nn as nn
import torchvision
from brats_dataset import BraTSDataset, BraTSDataset_3channel_input
import torch.backends.cudnn as cudnn

''''
Use:    This to investigate the discrimantor scores produces by VAAL and M-VAAL
        OR to plot the TSNE of the latent features

You should pass the arguments similar to main.py

Features:   Plot the discriminator scores.
            Plot TSNE plots 


'''

## Set Seed
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    cudnn.deterministic = True

def main(args):
    
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
        print (args.num_val, args.num_images)
        #args.budget = 300
        args.budget = 100
        args.initial_budget = 200
        args.num_classes = 2
        args.num_channels= 1

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
        # args.budget = 200
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

    elif args.dataset == 'brats_MIUA_HGG_3channel':
        
        crop = tuple(int(i) for i in args.crop.split(","))
        if min(crop) == 0:
            crop = None
        train_dataset =  BraTSDataset("data/BraTS_frames/MIUA/", flip = True, split_dir ="train/", resize= args.resize, crop = crop, version= 5,v_flip = True, brightness = True, rotation = True, random_crop= True, segmentation_type = args.segmentation_type)
        query_train_dataset = BraTSDataset_3channel_input("data/BraTS_frames/MIUA/", flip = False, split_dir ="train/", resize= 128, crop = (210,210), version= 5,v_flip = False, brightness = False, rotation = False, random_crop= False,segmentation_type = args.segmentation_type)
        test_dataset = BraTSDataset("data/BraTS_frames/MIUA/", flip = False, split_dir = "test/", resize= args.resize, crop = crop, version= 5,segmentation_type = args.segmentation_type)
        val_dataset = BraTSDataset("data/BraTS_frames/MIUA/", flip = False, split_dir = "val/", resize= args.resize, crop = crop, version= 5,segmentation_type = args.segmentation_type)
        
        args.num_val = len(val_dataset)
        args.num_images = len(train_dataset)
        print (args.num_val, args.num_images)
        args.budget = 100
        args.initial_budget = 200
        args.num_classes = 2
        args.num_channels= 1
        args.query_channels = 3

    else:
        raise NotImplementedError

    
    all_indices = np.arange(args.num_images)
    
    if args.train_full:
        initial_indices = all_indices.tolist()
    else:
        initial_indices = random.sample(list(all_indices), args.initial_budget)
    
    
    whole_dataloader = data.DataLoader(query_train_dataset,
            batch_size=args.batch_size, drop_last=False)
    

    args.device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu')
   

    splits = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    current_indices = list(initial_indices)
    for i, split in enumerate(splits):

        ## all unlabeled train samples
        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(query_train_dataset, 
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)

        

        if args.method == 'VAAL':
            #### initilaize the VAAL models
            discriminator = model.Discriminator(args.latent_dim)
            vae = model.VAE(args.latent_dim,nc=args.query_channels)

            # load the checkpoint models
            discriminator.load_state_dict(torch.load('./checkpoints/'+'/'+args.expt + 
                                        '/'+ 'discriminator_checkpoint'+str(split)+'.pth'))
            vae.load_state_dict(torch.load('./checkpoints/'+'/'+args.expt + 
                                        '/'+ 'vae_checkpoint'+str(split)+'.pth'))
 
            # send model to gpu
            discriminator = discriminator.to(device = args.device)
            vae = vae.to(device = args.device)
            
            VAAL_sampler = query_Sampler.AdversarySampler(args.budget)


            sampled_indices = VAAL_sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, unlabeled_indices,
                                             args.device)

        elif args.method == 'multimodal_VAAL':
            
            #### initilaize the VAAL models
            vae = multi_modal_model.VAE(args.latent_dim)
            discriminator = multi_modal_model.Discriminator(args.latent_dim)

            # load the checkpoint models
            discriminator.load_state_dict(torch.load('./checkpoints/'+'/'+args.expt + 
                                        '/'+ 'discriminator_checkpoint'+str(split)+'.pth'))
            vae.load_state_dict(torch.load('./checkpoints/'+'/'+args.expt + 
                                        '/'+ 'vae_checkpoint'+str(split)+'.pth'))

            # send the model to gpu
            vae = vae.to(device = args.device)
            discriminator = discriminator.to(device = args.device)

            

            multimodal_VAAL_sampler = query_Sampler.AdversarySampler_multimodal(args.budget)
            sampled_indices = multimodal_VAAL_sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, unlabeled_indices,
                                             args.device)

        elif args.method == "RandomSampling":
            
            random.seed(args.random_sampling_seed)
            random.shuffle(unlabeled_indices)
            sampled_indices = unlabeled_indices[:args.budget]

        old_indices = list(current_indices)
        
        
        # select the split that you want to investigate 
        if split == 0.05:
           
            #### for TSNE plots 
            '''
            
            if args.method == "RandomSampling":
                tsne(old_indices, sampled_indices, unlabeled_indices, whole_dataloader, vae = None, disc= None, method = "RandomSampling")
            elif args.method == "VAAL":
                tsne(old_indices, sampled_indices, unlabeled_indices, whole_dataloader, vae = vae, disc = discriminator, method = "VAAL")
            elif args.method == "multimodal_VAAL":
                tsne(old_indices, sampled_indices, unlabeled_indices, whole_dataloader, vae = vae, disc = discriminator, method = "multimodal_VAAL")

            '''


            #### for Discriminator Scores 
            if args.method == "VAAL":
                discriminator_hist(old_indices, sampled_indices, unlabeled_indices, whole_dataloader, vae = vae, disc = discriminator, method = "VAAL")
            elif args.method == "multimodal_VAAL":
                discriminator_hist(old_indices, sampled_indices, unlabeled_indices, whole_dataloader, vae = vae, disc = discriminator, method = "multimodal_VAAL")
            break

        
def extract_features (dataloader, model_name = "inception_v3", vae = None, disc = None):
    
    
    features_data = []

    if model_name == "inception_v3":
        model = torchvision.models.inception_v3(pretrained=True)
        model.fc = nn.Identity()
        model.to(args.device)
        model.eval()

    elif (model_name == "multimodalvae") or (model_name == "vae"):
        if  disc:
            model = vae
            model.eval()

            model_disc = disc
            model_disc.eval()
        else:
            model = vae
            model.eval()
            

    # put features and labels into arrays
    for batch_ix, (batch_image, batch_mask, batch_aux) in enumerate(dataloader):
        batch_image = batch_image.to(args.device)
        with torch.no_grad():
            if disc:
                if model_name =="multimodalvae":
                    _,_,_,batch_feature,_ = model(batch_image)
                elif model_name =="vae":
                     _,_,batch_feature,_ = model(batch_image)
            

                batch_feature = model_disc(batch_feature)
            else:
                if model_name == "inception_v3":
                    batch_feature = model(batch_image)
                elif model_name =="multimodalvae":
                    _,_,_,batch_feature,_ = model(batch_image)
                elif model_name =="vae":
                    _,_,batch_feature,_ = model(batch_image)
                   

        if disc:
            ft = batch_feature.flatten().cpu().numpy()
        else:
            ft = batch_feature.cpu().numpy()

       
        if disc:
            features_data.extend(ft)
        else:
            if batch_ix == 0:
                features_data = np.array(ft)
            np.concatenate((features_data,ft), axis = 0)

    if disc:
        features_data = np.array(features_data)
   
    return torch.Tensor(features_data)


def tsne(old_indices, sampled_indices, unlabeled_indices, whole_dataloader, vae = None, disc= None, method = "RandomSampling"):
    

    # whole_features= extract_features(whole_dataloader)
    
    if method == "VAAL":
        whole_features = extract_features(whole_dataloader, model_name= "vae", vae = vae, disc= None) 
    elif method == "multimodal_VAAL" :
        whole_features = extract_features(whole_dataloader, model_name= "multimodalvae", vae = vae,disc=None) 
    else:
        whole_features= extract_features(whole_dataloader)


    print (whole_features.shape)

    labels = ["unlabelled" for i in range (len(unlabeled_indices))]
    tsne = TSNE(n_components=2,n_iter=300)
    tsne_results = tsne.fit_transform(whole_features)
    print(len(tsne_results))
    tsne_X = tsne_results[:,0][unlabeled_indices]
    tsne_Y = tsne_results[:,1][unlabeled_indices]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_X, y=tsne_Y,
        hue= labels,
        data=tsne_results,
        legend="full",
        alpha=0.3)

    tsne_X = tsne_results[:,0][old_indices]
    tsne_Y = tsne_results[:,1][old_indices]
    labels = ["labelled" for i in range (len(old_indices))]

    sns.scatterplot(
        x=tsne_X, y=tsne_Y,
        hue= labels,
        data=tsne_results,
        legend="full",
        color=".2",
        palette="rocket",
        alpha=0.8)
    
    tsne_X = tsne_results[:,0][sampled_indices]
    tsne_Y = tsne_results[:,1][sampled_indices]
    labels = ["selected" for i in range (len(sampled_indices))]

    sns.scatterplot(
        x=tsne_X, y=tsne_Y,
        hue= labels,
        data=tsne_results,
        legend="full",
        color=".5",
        palette="viridis",
        alpha=0.8)
    
    save_name = "VAAL_TSNE.png"
    plt.savefig(save_name)
    plt.close(fig)

def discriminator_hist(old_indices, sampled_indices, unlabeled_indices, whole_dataloader, vae = None, disc= None, method = "VAAL"):

    if method == "VAAL":
        whole_features = extract_features(whole_dataloader, model_name= "vae", vae = vae, disc= disc) 
    elif method == "multimodal_VAAL" :
        whole_features = extract_features(whole_dataloader, model_name= "multimodalvae", vae = vae,disc= disc) 

    
    fig, ax = plt.subplots(figsize=(20, 15))
    matplotlib.rcParams.update({'font.size': 24})
    plt.subplot(2,1,1)
    plt.hist([i for i in whole_features[unlabeled_indices].tolist()], bins = 100, color = 'orange', label = "unlabelled data")
    plt.ylabel("No. of Samples")
    plt.xlabel("Discriminator Scores")
    plt.legend()
   
    plt.subplot(2,1,2)
    plt.hist([i for i in whole_features[old_indices].tolist()], bins = 100, label = "labelled data")
    plt.ylabel("No. of Samples")
    plt.xlabel("Discriminator Scores")
    plt.legend()
   
    save_name = "VAAL_disc_prob.png"
    plt.savefig(save_name)
    plt.close(fig)



if __name__ == '__main__':
    args = arguments.get_args()
    fix_seed(args.seed)
    main(args)

