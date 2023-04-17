import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()

    # general training related 
    parser.add_argument('--gpu_id', type =str , default= '0',help='select a gpu id')
    parser.add_argument('--dataset', type=str, default='liver-seg', help='Name of the dataset used.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size used for training and testing')
    parser.add_argument('--task_batch_size', type=int, default=32, help='Batch size used for training and testing')
    parser.add_argument('--query_train_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='use a random seed')
    parser.add_argument('--random_sampling_seed', type=int, default=0, help='use a random seed')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to where the data is')
    parser.add_argument('--with_replacement', action='store_true', default=False, help='replace the sample into into the unlabelled pool')
    

    # select the query strategy method
    parser.add_argument('--method', type=str, default="VAAL",  
                    choices=["RandomSampling", 
                             "LeastConfidence", 
                             "MarginSampling", 
                             "EntropySampling", 
                             "LeastConfidenceDropout", 
                             "MarginSamplingDropout", 
                             "EntropySamplingDropout", 
                             "KMeansSampling",
                             "KCenterGreedy", 
                             "BALDDropout", 
                             "VAAL",
                             "multimodal_VAAL"], help="query strategy")

    # VAAL related 
    parser.add_argument('--latent_dim', type=int, default=64, help='The dimensionality of the VAE latent dimension')
    parser.add_argument('--alpha1', type=float, default=1e-4, help='VAE learning rate')
    parser.add_argument('--alpha2', type=float, default=1e-4, help='discriminator learning rate')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--num_adv_steps', type=int, default=1, help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_vae_steps', type=int, default=2, help='Number of VAE steps taken for every task model step')
    parser.add_argument('--adversary_param', type=float, default=1, help='Hyperparameter for training. lambda2 in the paper')
    parser.add_argument('--mse_gamma1', type=float, default=1, help='Hyperparameter for training. Weight of RGB reconstriction error')
    parser.add_argument('--mse_gamma2', type=float, default=1, help='Hyperparameter for training. Weight of aux reconstruciton error')



    # general task related
    parser.add_argument('--task_type', type =str , default= 'segmentation', help='classification or segmentation task')
    parser.add_argument('--expt', type =str , default= 'Expt',
                    help='experiment name')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=5e-6,
                        help='Learning rate', dest='lr')
    parser.add_argument('--resize', type=int, default=0, help='fix_resize')
    parser.add_argument('--scale', type=str, default= "0,0", help='scale the image')
    parser.add_argument('--crop', type=str, default= "0,0", help='crop the image')
    parser.add_argument('--train_full', action ='store_true', default = False, help = "train the whole dataset")


    # segmentation task related
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--segmentation_type', type=str, default='whole_tumor', help='the type of segmentation task', choices= ["whole_tumor", "core_tumor", "ET"])
    
    # classification task related
    parser.add_argument('--architecture', default='resnet18', help='architecture used for classification')

    # save and log related 
    parser.add_argument('--out_path', type=str, default='./results', help='Path to where the output log will be')
    parser.add_argument('--log_name', type=str, default='accuracies.log', help='Final performance of the models will be saved with this name')

    
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    
    return args
