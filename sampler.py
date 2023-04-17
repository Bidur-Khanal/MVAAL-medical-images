import torch
import numpy as np

class AdversarySampler:
    def __init__(self, budget):
        self.budget = budget


    def sample(self, vae, discriminator, data, unlabeled_indices, device):

        vae.eval()
        discriminator.eval()
        all_preds = []

        for images, _,_  in data:
            images = images.to(device)

            with torch.no_grad():
                _, _, mu, _ = vae(images)

                #_, _, mu = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
          

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        print (all_preds)

        # select the points which the discriminator thinks are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = list(np.asarray(unlabeled_indices)[querry_indices])

        return querry_pool_indices


class AdversarySampler_multimodal:
    def __init__(self, budget):
        self.budget = budget


    def sample(self, vae, discriminator, data, unlabeled_indices, device):
        vae.eval()
        discriminator.eval()
        all_preds = []

        for images, _,_  in data:
            images = images.to(device)

            with torch.no_grad():
                _, _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
          

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator thinks are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = list(np.asarray(unlabeled_indices)[querry_indices])

        return querry_pool_indices
