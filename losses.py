import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    
class criterion_vgg(nn.Module):
    def __init__(self):
        super(criterion_vgg, self).__init__()
    
    def forward(self, w, x, y):  
        abs_diff = torch.abs(x - y)
        l1 = torch.mean(abs_diff, dim=[1,2,3]).unsqueeze(1)
        l1 = w*l1
        return l1
    
    
class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids
        
        self.loss_l1 = criterion_vgg()

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
            
        loss_f0 = self.loss_l1(self.weights[self.layids[0]], x_vgg[self.layids[0]], y_vgg[self.layids[0]].detach())
        loss_f1 = self.loss_l1(self.weights[self.layids[1]], x_vgg[self.layids[1]], y_vgg[self.layids[1]].detach())
        loss_f2 = self.loss_l1(self.weights[self.layids[2]], x_vgg[self.layids[2]], y_vgg[self.layids[2]].detach())
        loss_f3 = self.loss_l1(self.weights[self.layids[3]], x_vgg[self.layids[3]], y_vgg[self.layids[3]].detach())
        loss_f4 = self.loss_l1(self.weights[self.layids[4]], x_vgg[self.layids[4]], y_vgg[self.layids[4]].detach())
        
        loss = torch.cat([loss_f0, loss_f1, loss_f2, loss_f3, loss_f4], 1)
        return loss
    
    
def GMMLoss(parse_cloth, warp_coarse_cloth, warp_fine_cloth):
    
    loss_l1 = nn.L1Loss()
    loss_vgg = VGGLoss()
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    lambda1 = 1
    lambda2 = 1
    lambda3 = 1
    lambda4 = 0.5
    lambda5 = 0.5
    k = 3
    
    ls0 = loss_l1(parse_cloth, warp_coarse_cloth)
    ls1 = loss_l1(parse_cloth, warp_fine_cloth)
    
    lpush = k*ls1 - loss_l1(warp_fine_cloth, warp_coarse_cloth)
    
    v0 = loss_vgg(warp_coarse_cloth, parse_cloth)
    v1 = loss_vgg(warp_fine_cloth, parse_cloth)
    cos = torch.mean(cos_sim(v0, v1))
    lalign = torch.pow(cos-1, 2)
    
    lpgm = lambda4*lpush + lambda5*lalign
    loss = lambda1*ls0 + lambda2*ls1 + lambda3*lpgm
    
    return loss


def segm_unet_loss(output, target):
    
    w = 1.5
    weights = np.array([w,w,w,1,w,1,1,1,1,w,1,1,1,w,w,1,1,1,1,1], dtype=np.float32)
    weights = torch.from_numpy(weights).to(device)
    
    loss_ce = nn.CrossEntropyLoss(weight=weights)
    loss = loss_ce(output, target)
    
    return loss


def tom_loss(img_tryon, mask_c, img_model, mask_pc, img_tryon_tri=None):
    loss_l1 = nn.L1Loss()
    loss_vgg = VGGLoss()
    # loss_bce = nn.BCELoss()

    loss_p = loss_l1(img_tryon, img_model)
    loss_mask = loss_l1(mask_c, mask_pc)
    loss_perc = torch.mean(loss_vgg(img_tryon, img_model))
    # loss_mask_ce = loss_bce(mask_c, mask_pc)

    l_tt = loss_p + loss_perc + loss_mask

    if img_tryon_tri is not None:
        d_neg = loss_l1(img_tryon, img_tryon_tri)
        d_pov = loss_l1(img_tryon, img_model)
        loss_d = max(d_pov-d_neg, 0)
        return(l_tt + loss_d)
    else:
        return(l_tt)