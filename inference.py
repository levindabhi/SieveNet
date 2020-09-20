import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
import json
import os
import os.path as osp
from PIL import Image
from PIL import ImageDraw

import time
import warnings
from tqdm import tqdm
from predict_pose import generate_pose_keypoints
from visualization import load_checkpoint, save_images
from gmm import GMM
from unet import UnetGenerator
from config import parser

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensortoimage(t, path):
    im = transforms.ToPILImage()(t).convert("RGB")
    im.save(path)

def generate_data(opt, im_path, cloth_path, pose_path, segm_path):

    transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5,), (0.5,))])

    c = Image.open(cloth_path)
    c = transform(c)

    im = Image.open(osp.join(im_path))
    im = transform(im)

    im_parse = Image.open(segm_path)
    parse_array = np.array(im_parse)

    parse_shape = (parse_array > 0).astype(np.float32)

    parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)

    parse_ttp = (parse_array == 1).astype(np.float32) + \
        		(parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32) + \
                (parse_array == 3).astype(np.float32) + \
                (parse_array == 8).astype(np.float32) + \
                (parse_array == 9).astype(np.float32) + \
                (parse_array == 10).astype(np.float32) + \
                (parse_array == 11).astype(np.float32) + \
                (parse_array == 12).astype(np.float32) + \
                (parse_array == 14).astype(np.float32) + \
                (parse_array == 3).astype(np.float32) + \
                (parse_array >= 15).astype(np.float32)
    
    phead = torch.from_numpy(parse_head)
    ptexttp = torch.from_numpy(parse_ttp)

    # shape downsample
    parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
    parse_shape = parse_shape.resize((opt.fine_width//16, opt.fine_height//16), Image.BILINEAR)
    parse_shape = parse_shape.resize((opt.fine_width, opt.fine_height), Image.BILINEAR)
    shape = transform(parse_shape) # [-1,1]

    im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts
    im_ttp = im * ptexttp - (1- ptexttp)

    with open(pose_path, 'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1,3))

    point_num = pose_data.shape[0]
    pose_map = torch.zeros(point_num, opt.fine_height, opt.fine_width)
    r = opt.radius
    im_pose = Image.new('L', (opt.fine_width, opt.fine_height))
    pose_draw = ImageDraw.Draw(im_pose)
    for i in range(point_num):
        one_map = Image.new('L', (opt.fine_width, opt.fine_height))
        draw = ImageDraw.Draw(one_map)
        pointx = pose_data[i,0]
        pointy = pose_data[i,1]
        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
        one_map = transform(one_map)
        pose_map[i] = one_map[0]
    
    # cloth-agnostic representation
    agnostic = torch.cat([shape, im_h, pose_map], 0)

    return(torch.unsqueeze(agnostic, 0), torch.unsqueeze(c,0), torch.unsqueeze(im_ttp,0))

def main():
    opt = parser()

    im_path = opt.input_image_path                                               #person image path
    cloth_path =  opt.cloth_image_path                                           #cloth image path
    pose_path = opt.input_image_path.replace('.jpg', '_keypoints.json')          #pose keypoint path
    generate_pose_keypoints(im_path)                                             #generating pose keypoints
    segm_path = opt.human_parsing_image_path                                     #segemented mask path
    img_name = im_path.split('/')[-1].split('.')[0] + '_' 

    agnostic, c, im_ttp = generate_data(opt, im_path, cloth_path, pose_path, segm_path)
    
    agnostic = agnostic.to(device)
    c = c.to(device)
    im_ttp = im_ttp.to(device)

    gmm = GMM(opt)
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, 'GMM', 'gmm_final.pth'))
    gmm.to(device)
    gmm.eval()

    unet_mask = UnetGenerator(25, 20, ngf=64)
    load_checkpoint(unet_mask, os.path.join(opt.checkpoint_dir, 'SEG', 'segm_final.pth'))
    unet_mask.to(device)
    unet_mask.eval()

    tom = UnetGenerator(26, 4, ngf=64)
    load_checkpoint(tom, os.path.join(opt.checkpoint_dir, 'TOM', 'tom_final.pth'))
    tom.to(device)
    tom.eval()

    with torch.no_grad():
        output_segm = unet_mask(torch.cat([agnostic, c], 1))
        grid_zero, theta, grid_one, delta_theta = gmm(agnostic, c)
    c_warp = F.grid_sample(c, grid_one, padding_mode='border')
    output_segm = F.log_softmax(output_segm, dim=1)

    output_argm = torch.max(output_segm, dim=1, keepdim=True)[1]
    final_segm = torch.zeros(output_segm.shape).to(device).scatter(1, output_argm, 1.0)

    input_tom = torch.cat([final_segm, c_warp, im_ttp], 1)

    with torch.no_grad():
        output_tom = tom(input_tom)
    person_r = torch.tanh(output_tom[:,:3,:,:])
    mask_c = torch.sigmoid(output_tom[:,3:,:,:])
    mask_c = (mask_c >= 0.5).type(torch.float)
    img_tryon = mask_c * c_warp + (1 - mask_c) * person_r
    print('Output generated!')

    c_warp = c_warp*0.5+0.5
    output_argm = output_argm.type(torch.float)
    person_r = person_r*0.5+0.5
    img_tryon = img_tryon*0.5+0.5

    tensortoimage(c_warp[0].cpu(), osp.join(opt.save_dir, img_name+'w_cloth.png'))
    tensortoimage(output_argm[0][0].cpu(), osp.join(opt.save_dir, img_name+'seg_mask.png'))
    tensortoimage(mask_c[0].cpu(), osp.join(opt.save_dir, img_name+'c_mask.png'))
    tensortoimage(person_r[0].cpu(), osp.join(opt.save_dir, img_name+'ren_person.png'))
    tensortoimage(img_tryon[0].cpu(), osp.join(opt.save_dir, img_name+'final_output.png'))
    print('Output saved at {}'.format(opt.save_dir))

if __name__ == "__main__": 
    main()