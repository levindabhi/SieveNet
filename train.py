import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os.path as osp
import os
import time
import warnings

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

def train_gmm(opt, train_loader, model, board):
    
    print('----Traning of module {} started----'.format(opt.name))
    model.to(device)
    model.train()
    
    #used scheduler for training of gmm
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    lr_lambda = lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_lambda)
    
    
    for step in range(opt.previous_step, opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].to(device)
        im_h = inputs['head'].to(device)
        agnostic = inputs['agnostic'].to(device)
        c = inputs['cloth'].to(device)
        im_c =  inputs['parse_cloth'].to(device)
        im_g = inputs['grid_image'].to(device)
        
        grid_zero, theta, grid_one, delta_theta = model(agnostic, c)
        
        warped_coarse_cloth = F.grid_sample(c, grid_zero, padding_mode='border')
        warped_cloth = F.grid_sample(c, grid_one, padding_mode='border')
        warped_coarse_grid = F.grid_sample(im_g, grid_zero, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid_one, padding_mode='zeros')

        visuals = [[im_h, im, (warped_cloth+im)*0.5],
                   [warped_grid, warped_coarse_grid, warped_grid],
                   [im_c, warped_coarse_cloth, warped_cloth]]
        
        loss = GMMLoss(im_c, warped_coarse_cloth, warped_cloth)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('loss', loss.item(), step+1)
            board.add_scalar('GPU Allocated', round(torch.cuda.memory_allocated(0)/1024**3,1), step+1)
            board.add_scalar('GPU Cached', round(torch.cuda.memory_cached(0)/1024**3,1), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))
            
            
            
            
def train_segm(opt, train_loader, model, board):

    print('----Traning of module {} started----'.format(opt.name))
    model.to(device)
    model.train()
    
    #used scheduler for training seg mask generation module
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    lr_lambda = lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_lambda)

    for step in range(opt.previous_step, opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].to(device)
        agnostic = inputs['agnostic'].to(device)
        c = inputs['cloth'].to(device)
        im_parse = inputs['parse_model'].to(device)


        input_unet = torch.cat([agnostic, c], 1)
        output_unet = model(input_unet)

        #argmax
        output_segm = F.log_softmax(output_unet, dim=1)
        #generating 20 mask channel from argmax
        output_segm = torch.max(output_segm, dim=1, keepdim=True)[1]

        im_parse = im_parse.type(torch.long)   
        loss_unet = segm_unet_loss(output_unet, im_parse) 
        optimizer.zero_grad()
        loss_unet.backward()
        optimizer.step()

        #tensorboard is showing binary image
        visuals = [[torch.unsqueeze(im_parse, dim=1), output_segm]]
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('loss', loss_unet.item(), step+1)
            board.add_scalar('GPU Allocated', round(torch.cuda.memory_allocated(0)/1024**3,1), step+1)
            board.add_scalar('GPU Cached', round(torch.cuda.memory_cached(0)/1024**3,1), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss of unet: %4f' % (step+1, t, loss_unet.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))



def train_tom(opt, train_loader, model, model_triloss, board):

    print('----Traning of module {} started----'.format(opt.name))
    model.to(device)
    model.train()

    unet_mask = UnetGenerator(25, 20, ngf=64)
    load_checkpoint(unet_mask, os.path.join(opt.checkpoint_dir, 'SEG', 'segm_final.pth'))
    unet_mask.to(device)
    unet_mask.eval()

    gmm = GMM(opt)
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, 'GMM', 'gmm_final.pth'))
    gmm.to(device)
    gmm.eval()

    #scheduler is not used during training
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    # lr_lambda = lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_lambda)
    
    #for updating network parameter used for triplet loss
    update_checkpoint = None

    for step in range(opt.previous_step, opt.keep_step + opt.decay_step):

        iter_start_time = time.time()
        inputs = train_loader.next_batch()
        
        im_h = inputs['head'].to(device)    
        im = inputs['image'].to(device)
        agnostic = inputs['agnostic'].to(device)
        c = inputs['cloth'].to(device)
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

        output_tom = model(input_tom)
        person_r = torch.tanh(output_tom[:,:3,:,:])
        mask_c = torch.sigmoid(output_tom[:,3:,:,:])
        img_tryon = mask_c * c_warp + (1 - mask_c) * person_r

        #if steps greater than k then adding triplet to main tom loss
        if step >= opt.keep_step:
            model_triloss.to(device)
            model_triloss.eval()
        
        if step > opt.keep_step:
            i_prev = int(step/opt.save_count)*opt.save_count

            if update_checkpoint!=i_prev:
                path = osp.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (i_prev))
                load_checkpoint(model_triloss, path)
                update_checkpoint = i_prev
            
            with torch.no_grad():
                output_tom_tri = model_triloss(input_tom)
            person_r_tri = torch.tanh(output_tom_tri[:,:3,:,:])
            mask_c_tri = torch.sigmoid(output_tom_tri[:,3:,:,:])
            img_tryon_tri = mask_c_tri * c_warp + (1 - mask_c_tri) * person_r_tri
            loss = tom_loss(img_tryon, mask_c, im, im_c_mask, img_tryon_tri)
        else:
            loss = tom_loss(img_tryon, mask_c, im, im_c_mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        visuals = [[im, c, img_tryon], 
                    [im_c, c_warp, person_r],
                    [im_c_mask,mask_c, im_h]]

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('loss', loss.item(), step+1)
            board.add_scalar('GPU Allocated', round(torch.cuda.memory_allocated(0)/1024**3,1), step+1)
            board.add_scalar('GPU Cached', round(torch.cuda.memory_cached(0)/1024**3,1), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))
            
            
            
def main():
    opt = parser()

    train_dataset = SieveDataset(opt)
    # create dataloader
    train_loader = SieveDataLoader(opt, train_dataset)
   
    # create model & train & save the final checkpoint
    if opt.name == 'GMM':
        model = GMM(opt)
        
        # visualization
        if not os.path.exists(os.path.join(opt.tensorboard_dir, opt.name)):
            os.makedirs(os.path.join(opt.tensorboard_dir, opt.name))
        board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
    
        if not os.path.exists(os.path.join(opt.checkpoint_dir, opt.name)):
            os.makedirs(os.path.join(opt.checkpoint_dir, opt.name))
            
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
        
    elif opt.name == 'SEG':
        
        #input channel = agnostic(22) + cloth(3)
        #output channel = segemantion output(20)
        model = UnetGenerator(25, 20, ngf=64)

        # visualization
        if not os.path.exists(os.path.join(opt.tensorboard_dir, opt.name)):
            os.makedirs(os.path.join(opt.tensorboard_dir, opt.name))
        board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))

        #for checkpoint saving
        if not os.path.exists(os.path.join(opt.checkpoint_dir, opt.name)):
            os.makedirs(os.path.join(opt.checkpoint_dir, opt.name))

        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
            print('Checkpoints loaded!')
        train_segm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'segm_final.pth'))


    elif opt.name == 'TOM':

        #input channel = generated seg mask(20) + texture translation prior(3) + warped cloth(3)
        #output channel = cloth mask(1) + rendered person(3)
        model = UnetGenerator(26, 4, ngf=64)
        
        #for  Duelling Triplet Loss Strategy
        model_triloss = UnetGenerator(26, 4, ngf=64)

        # visualization
        if not os.path.exists(os.path.join(opt.tensorboard_dir, opt.name)):
            os.makedirs(os.path.join(opt.tensorboard_dir, opt.name))
        board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))

        if not os.path.exists(os.path.join(opt.checkpoint_dir, opt.name)):
            os.makedirs(os.path.join(opt.checkpoint_dir, opt.name))

        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, model_triloss, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))

      
        
if __name__ == '__main__':
    main()
    