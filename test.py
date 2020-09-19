import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os.path as osp
import os
import time
import warnings
from tqdm import tqdm

from visualization import board_add_image, board_add_images, load_checkpoint, save_checkpoint, save_images
from dataset import SieveDataset, SieveDataLoader
from gmm import GMM
from unet import UnetGenerator
from losses import GMMLoss, segm_unet_loss, tom_loss
from config import parser

import sys
torch.set_printoptions(profile="full")

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_gmm(opt, test_loader, model, board):
    
    print('----Testing of module {} started----'.format(opt.name))
    model.to(device)
    model.eval()
    
    length = len(test_loader.data_loader)
    step = 0
    pbar = tqdm(total=length)

    inputs = test_loader.next_batch()
    while inputs is not None:
        im_name = inputs['im_name']
        im = inputs['image'].to(device)
        im_h = inputs['head'].to(device)
        agnostic = inputs['agnostic'].to(device)
        c = inputs['cloth'].to(device)
        im_c =  inputs['parse_cloth'].to(device)
        im_g = inputs['grid_image'].to(device)

        with torch.no_grad():
            grid_zero, theta, grid_one, delta_theta = model(agnostic, c)
        
        warped_coarse_cloth = F.grid_sample(c, grid_zero, padding_mode='border')
        warped_cloth = F.grid_sample(c, grid_one, padding_mode='border')
        warped_coarse_grid = F.grid_sample(im_g, grid_zero, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid_one, padding_mode='zeros')

        visuals = [[im_h, im, (warped_cloth+im)*0.5],
                   [warped_grid, warped_coarse_grid, warped_grid],
                   [im_c, warped_coarse_cloth, warped_cloth]]
        board_add_images(board, 'combine', visuals, step+1)
        save_images(warped_cloth,im_name, osp.join(opt.dataroot, opt.datamode, 'wrap-cloth'))

        inputs = test_loader.next_batch()
        step+=1
        pbar.update(1)
        
def test_tom(opt, test_loader, model, board):
    print('----Testing of module {} started----'.format(opt.name))
    model.to(device)
    model.eval()

    unet_mask = UnetGenerator(25, 20, ngf=64)
    load_checkpoint(unet_mask, os.path.join(opt.checkpoint_dir, 'SEG', 'segm_final.pth'))
    unet_mask.to(device)
    unet_mask.eval()

    gmm = GMM(opt)
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, 'GMM', 'gmm_final.pth'))
    gmm.to(device)
    gmm.eval()
    
    length = len(test_loader.data_loader)
    step = 0
    pbar = tqdm(total=length)

    inputs = test_loader.next_batch()
    while inputs is not None:
        im_name = inputs['im_name']
        im_h = inputs['head'].to(device)    
        im = inputs['image'].to(device)
        agnostic = inputs['agnostic'].to(device)
        c = inputs['cloth'].to(device)
        # c_warp = inputs['cloth_warp'].to(device)
        im_c =  inputs['parse_cloth'].to(device)
        im_c_mask = inputs['parse_cloth_mask'].to(device)
        im_ttp = inputs['texture_t_prior'].to(device)

        with torch.no_grad():
            output_segm = unet_mask(torch.cat([agnostic, c], 1))
            grid_zero, theta, grid_one, delta_theta = gmm(agnostic, c)
        c_warp = F.grid_sample(c, grid_one, padding_mode='border')
        output_segm = F.log_softmax(output_segm, dim=1)

        output_argm = torch.max(output_segm, dim=1, keepdim=True)[1]
        final_segm = torch.zeros(output_segm.shape).to(device).scatter(1, output_argm, 1.0)
        input_tom = torch.cat([final_segm, c_warp, im_ttp], 1)

        with torch.no_grad():
            output_tom = model(input_tom)
        person_r = torch.tanh(output_tom[:,:3,:,:])
        mask_c = torch.sigmoid(output_tom[:,3:,:,:])
        mask_c = (mask_c >= 0.5).type(torch.float)
        img_tryon = mask_c * c_warp + (1 - mask_c) * person_r

        visuals = [[im, c, img_tryon], 
                [im_c, c_warp, person_r],
                [im_c_mask,mask_c, im_h]]
        board_add_images(board, 'combine', visuals, step+1)
        save_images(img_tryon,im_name, osp.join(opt.dataroot, opt.datamode, 'final-output'))

        inputs = test_loader.next_batch()
        step+=1
        pbar.update(1)



def main():
    opt = parser()

    test_dataset = SieveDataset(opt)
    # create dataloader
    test_loader = SieveDataLoader(opt, test_dataset)
   
   
    if opt.name == 'GMM':
        model = GMM(opt)
        
        # visualization
        if not os.path.exists(os.path.join(opt.tensorboard_dir, opt.name, opt.datamode)):
            os.makedirs(os.path.join(opt.tensorboard_dir, opt.name, opt.datamode))
        board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name, opt.datamode))
            
        checkpoint_path = osp.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth')
        load_checkpoint(model, checkpoint_path)
        test_gmm(opt, test_loader, model, board)


    elif opt.name == 'TOM':
        model = UnetGenerator(26, 4, ngf=64)
        
        # visualization
        if not os.path.exists(os.path.join(opt.tensorboard_dir, opt.name, opt.datamode)):
            os.makedirs(os.path.join(opt.tensorboard_dir, opt.name, opt.datamode))
        board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name, opt.datamode))

        checkpoint_path = osp.join(opt.checkpoint_dir, opt.name, 'tom_final.pth')
        load_checkpoint(model, checkpoint_path)
        test_tom(opt, test_loader, model, board)
  
if __name__ == '__main__':
    main()
    