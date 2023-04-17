import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path
import sampler
import copy
from torch.autograd import grad as torch_grad
from torch.autograd import Variable
from torch import autograd




class multi_modal_VAAL_Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader
        self.device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu')
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dir_checkpoint = Path('./checkpoints/')
        self.sampler = sampler.AdversarySampler_multimodal(self.args.budget)
        self.gp_weight =1


    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, aux in dataloader:
                    yield img, aux, label
        else:
            while True:
                for img, _, aux in dataloader:
                    yield img, aux

    def _gradient_penalty(self, disc, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)

        #print (alpha.shape,real_data.shape,generated_data.shape)
        
        alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        
        interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = disc(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                                grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                                create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()


    def train(self, current_split, querry_dataloader, val_dataloader, vae, discriminator, unlabeled_dataloader):
        self.args.train_iterations = (self.args.num_images* self.args.query_train_epochs) // self.args.batch_size
       
        
        if not (os.path.exists(str(self.dir_checkpoint)+'/'+self.args.expt + '/'+ 'vae_checkpoint'+str(current_split)+'.pth') and os.path.exists(str(self.dir_checkpoint)+'/'+self.args.expt + '/'+ 'discriminator_checkpoint'+str(current_split)+'.pth')):
            labeled_data = self.read_data(querry_dataloader)
            unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

            optim_vae = optim.Adam(vae.parameters(), lr=self.args.alpha1)
            optim_discriminator = optim.Adam(discriminator.parameters(), lr=self.args.alpha2)

            vae.train()
            discriminator.train()

        
            
            for iter_count in range(self.args.train_iterations):
                
                labeled_imgs, labeled_aux, labels = next(labeled_data)
                unlabeled_imgs, unlabeled_aux = next(unlabeled_data)
            
                labeled_imgs = labeled_imgs.to(device=self.args.device, dtype=torch.float32)
                labeled_aux = labeled_aux.to(device=self.args.device, dtype=torch.float32)

                unlabeled_imgs = unlabeled_imgs.to(device=self.args.device, dtype=torch.float32)
                unlabeled_aux = unlabeled_aux.to(device=self.args.device, dtype=torch.float32)
                labels = labels.to(device=self.args.device, dtype=torch.long)


                # VAE step
                for count in range(self.args.num_vae_steps):
                    recon, aux_recon, z, mu, logvar = vae(labeled_imgs)
                
                    unsup_loss = self.vae_loss(labeled_imgs, labeled_aux, recon, aux_recon, mu, logvar, self.args.beta)
                    unlab_recon, unlab_aux_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)

                    
                    transductive_loss = self.vae_loss(unlabeled_imgs, unlabeled_aux,
                            unlab_recon, unlab_aux_recon, unlab_mu, unlab_logvar, self.args.beta)
                
                    labeled_preds = discriminator(mu)
                    unlabeled_preds = discriminator(unlab_mu)
                    
                    lab_real_preds = torch.ones(labeled_imgs.size(0))
                    unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                        

                    lab_real_preds = lab_real_preds.to(device=self.args.device)
                    unlab_real_preds = unlab_real_preds.to(device=self.args.device)


                    # dsc_loss = self.bce_loss(labeled_preds[:,0], lab_real_preds) + \
                    #         self.bce_loss(unlabeled_preds[:,0], unlab_real_preds)
                    
                    # change to GANGP
                    real_loss = labeled_preds.mean()
                    fake_loss = unlabeled_preds.mean()

                    # gradient_penalty = self._gradient_penalty(mu.data, unlab_mu.data)
                    dsc_loss = -fake_loss + real_loss 


                    total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
                    optim_vae.zero_grad()
                    total_vae_loss.backward()
                    optim_vae.step()

                    # sample new batch if needed to train the adversarial network
                    if count < (self.args.num_vae_steps - 1):
                        labeled_imgs, labeled_aux, _ = next(labeled_data)
                        unlabeled_imgs, unlabeled_aux = next(unlabeled_data)

                        labeled_imgs = labeled_imgs.to(device=self.args.device, dtype=torch.float32)
                        labeled_aux = labeled_aux.to(device=self.args.device, dtype=torch.float32)

                        unlabeled_imgs = unlabeled_imgs.to(device=self.args.device, dtype=torch.float32)
                        unlabeled_aux = unlabeled_aux.to(device=self.args.device, dtype=torch.float32)
                        labels = labels.to(device=self.args.device, dtype=torch.long)

                # Discriminator step
                for count in range(self.args.num_adv_steps):
                    with torch.no_grad():
                        _, _, _, mu, _ = vae(labeled_imgs)
                        _, _, _, unlab_mu, _ = vae(unlabeled_imgs)
                    
                    labeled_preds = discriminator(mu)
                    unlabeled_preds = discriminator(unlab_mu)
                    
                    lab_real_preds = torch.ones(labeled_imgs.size(0))
                    unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                    
                    lab_real_preds = lab_real_preds.to(device=self.args.device)
                    unlab_fake_preds = unlab_fake_preds.to(device=self.args.device)
                    
                    # dsc_loss = self.bce_loss(labeled_preds[:,0], lab_real_preds) + \
                    #         self.bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

                    # change to GANGP
                    real_loss = labeled_preds.mean()
                    fake_loss = unlabeled_preds.mean()

                    gradient_penalty = self._gradient_penalty(discriminator, mu, unlab_mu)
                    dsc_loss = fake_loss - real_loss +  gradient_penalty

                    optim_discriminator.zero_grad()
                    dsc_loss.backward()
                    optim_discriminator.step()

                    # sample new batch if needed to train the adversarial network
                    if count < (self.args.num_adv_steps - 1):
                        labeled_imgs, labeled_aux, _ = next(labeled_data)
                        unlabeled_imgs, unlabeled_aux = next(unlabeled_data)

                        labeled_imgs = labeled_imgs.to(device=self.args.device, dtype=torch.float32)
                        labeled_aux = labeled_aux.to(device=self.args.device, dtype=torch.float32)
                        unlabeled_imgs = unlabeled_imgs.to(device=self.args.device, dtype=torch.float32)
                        unlabeled_aux = unlabeled_aux.to(device=self.args.device, dtype=torch.float32)
                        labels = labels.to(device=self.args.device, dtype=torch.long)

                    

                if iter_count % 100 == 0:
                    print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                    print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))


            Path(str(self.dir_checkpoint)+'/'+self.args.expt+'/multi_modal_VAAL').mkdir(parents=True, exist_ok=True)
            torch.save(vae.state_dict(), str(self.dir_checkpoint)+'/'+self.args.expt + '/'+ 'vae_checkpoint'+str(current_split)+'.pth')
            torch.save(discriminator.state_dict(), str(self.dir_checkpoint)+'/'+self.args.expt + '/'+ 'discriminator_checkpoint'+str(current_split)+'.pth')
        else:
            # load the checkpoint models
            discriminator.load_state_dict(torch.load(str(self.dir_checkpoint)+'/'+self.args.expt + '/'+ 'discriminator_checkpoint'+str(current_split)+'.pth'))
            vae.load_state_dict(torch.load(str(self.dir_checkpoint)+'/'+self.args.expt + '/'+ 'vae_checkpoint'+str(current_split)+'.pth'))
            
        return vae, discriminator


    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader, unlabeled_indices):
       
        querry_indices = self.sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, unlabeled_indices,
                                             self.args.device)

        return querry_indices
                


    def vae_loss(self, x, aux,recon, aux_recon, mu, logvar, beta):
        MSE1 = self.mse_loss(recon, x)
        MSE2 = self.mse_loss(aux_recon,aux)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        MSE = self.args.mse_gamma1*MSE1 + self.args.mse_gamma2*MSE2
        return  MSE + KLD

