import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

from model_custom.lenet import LeNet


def safe_load_dict(model, new_model_state):
    """
    Safe loading of previous ckpt file.
    """
    old_model_state = model.state_dict()
    c = 0
    for name, param in new_model_state.items():
        n = name.split('.')
        beg = n[0]
        end = n[1:]
        if beg == 'model':
            name = '.'.join(end)
        if name not in old_model_state:
            print('%s not found in old model.' % name)
            continue
        c += 1
        if old_model_state[name].shape != param.shape:
            print('Shape mismatch...ignoring %s' % name)
            continue
        else:
            old_model_state[name].copy_(param)
    if c == 0:
        raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')


class ResNet18_Freeze_Upto(nn.Module):
    def __init__(self, pretrained = False,ckpt_file = None, unfreeze_layer= None, num_classes=None):
        super(ResNet18_Freeze_Upto, self).__init__()

        self.model = models.resnet18(pretrained=pretrained)
        if num_classes is not None:
                print('Changing output layer to contain %d classes.' % num_classes)
                self.model.fc = nn.Linear(512, num_classes)

        if ckpt_file is not None:
            resumed = torch.load(ckpt_file)
        
            if 'state_dict' in resumed:
                state_dict_key = 'state_dict'
                print("Resuming from {}".format(ckpt_file))
                safe_load_dict(self.model, resumed[state_dict_key])
            else:
                print("Resuming from {}".format(ckpt_file))
                safe_load_dict(self.model, resumed)
       
        for name, param in self.model.named_parameters():
            
            if (param.requires_grad) and (not any( layer in name for layer in unfreeze_layer)):
                param.requires_grad = False
                
           
    def forward(self, x):
        out = self.model(x)
        return out



class LeNet_Freeze_Upto(nn.Module):
    def __init__(self, ckpt_file = None, unfreeze_layer= None, num_classes=None):
        super(LeNet_Freeze_Upto, self).__init__()

        self.model = LeNet(num_classes=num_classes)
        if num_classes is not None:
                print('Changing output layer to contain %d classes.' % num_classes)
                self.model.fc3 = nn.Linear(84, num_classes)

        if ckpt_file is not None:
            resumed = torch.load(ckpt_file)

            if 'state_dict' in resumed:
                state_dict_key = 'state_dict'
                print("Resuming from {}".format(ckpt_file))
                safe_load_dict(self.model, resumed[state_dict_key])
            else:
                print("Resuming from {}".format(ckpt_file))
                safe_load_dict(self.model, resumed)

        if unfreeze_layer is not None:
            
            for name, param in self.model.named_parameters():
                
                if (param.requires_grad) and (not any( layer in name for layer in unfreeze_layer)):
                    param.requires_grad = False
                    
           
    def forward(self, x):
        out = self.model(x)
        return out

if __name__ == "__main__":
    layer_name = ["layer4.1","fc.weight", "fc.bias"]
    model = ResNet18_Freeze_Upto( unfreeze_layer=layer_name)
    
    