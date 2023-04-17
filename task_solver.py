import logging
import sys
from pathlib import Path
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import numpy as np
from dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
import pdb

import warnings
warnings.filterwarnings("ignore")

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_dice = 0.

    def early_stop(self, validation_dice):
        if validation_dice > self.min_dice:
            self.min_dice = validation_dice
            self.counter = 0
        elif validation_dice < (self.min_dice + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_task(args,net, train_loader, val_loader, test_loader,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              save_checkpoint: bool = True,
              amp: bool = False, wandb_log = None, split = 1):


    dir_checkpoint = Path('./checkpoints/')
    best_val_dice = 0.
   
    
    # 4. Set up the optimizer, the loss, the learning rate scheduler 
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=7)  # goal: maximize Dice score

    if net.n_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    global_step = 0
    

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader)*args.batch_size, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for images, true_masks, aux in train_loader:

             
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=args.device, dtype=torch.float32)
                true_masks = true_masks.to(device=args.device, dtype=torch.long)
                

                masks_pred = net(images)
                #pdb.set_trace()
               
                if net.n_classes == 1:

                    loss = criterion(masks_pred, torch.unsqueeze(true_masks,dim = 1).float()) + dice_loss((F.sigmoid(masks_pred) > 0.5).float(),torch.unsqueeze(true_masks,dim = 1).float(),
                                        multiclass=False)

                else:
                    loss = criterion(masks_pred, true_masks) \
                            + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                        multiclass=True)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                wandb_log.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        ### Evaluation round
        val_score = evaluate(net, val_loader, args.device, ignore_background= True)
        scheduler.step(val_score)

        logging.info('Validation Dice score: {}'.format(val_score))

        if net.n_classes == 1:
            wandb_mask  = (F.sigmoid(masks_pred) > 0.5).float().cpu()
        else:
            wandb_mask = masks_pred.argmax(dim=1)[0].float().cpu()
        wandb_log.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'validation Dice': val_score,
            'images': wandb.Image(images[0].cpu()),
            'masks': {
                'true': wandb.Image(true_masks[0].float().cpu()),
                'pred': wandb.Image(wandb_mask[0].float().cpu()),
            },
            'step': global_step,
            'epoch': epoch
        })       

        if best_val_dice < val_score:
            best_val_dice = val_score
            if save_checkpoint:
                Path(str(dir_checkpoint)+'/'+args.expt).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint)+'/'+args.expt + '/'+ 'checkpoint'+str(split)+'.pth')
                logging.info(f'Checkpoint {epoch} saved!')

            

    net.load_state_dict(torch.load(str(dir_checkpoint)+'/'+args.expt + '/'+ 'checkpoint'+str(split)+'.pth'))
    test_score = evaluate(net, test_loader, args.device, ignore_background= True)
    wandb_log.log({'test Dice': test_score})
    
