import numpy as np
import json
import os
import os.path as osp
from PIL import Image
from PIL import ImageDraw

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class SieveDataset(data.Dataset):
    """
        Dataset for SieveNet.
    """
    def __init__(self, opt):
        super(SieveDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode 
        self.stage = opt.stage 
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode) 
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5,), (0.5,))])
        
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "SieveDataset"

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # cloth image & cloth mask
        #sievenet is not using cloth mask in any stage
        
        c = Image.open(osp.join(self.data_path, 'cloth', c_name))
        c = self.transform(c)  # [-1,1]

        # person image 
        im = Image.open(osp.join(self.data_path, 'image', im_name))
        im = self.transform(im) # [-1,1]

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))

        '''
        0: backgrounnd, 1: hat, 2: hair, 3: glove, 4: sunglasses, 5: upper-clothers,
        6: dress, 7: coat, 8: socks, 9: pants, 10: jumpsiuts,
        11: scarf, 12: skirt, 13: face, 14: left-arm, 15: right-arm,
		16: left-leg, 17: right-leg, 18: left shoe, 19 right shoe
        '''
        
        parse_array = np.array(im_parse)
        
        #shape of person
        parse_shape = (parse_array > 0).astype(np.float32)

        #head of person
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)

        #cloth person is wearing
        #here try-on is of upper cloth
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32)
        
        #background in image of person
        parse_background = (parse_array == 0).astype(np.float32)
        
        #texture translation prior required in last stage
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
        
        ptexttp = torch.from_numpy(parse_ttp)
        im_ttp = im * ptexttp - (1- ptexttp) # [-1,1], fill 0 for other parts
        
        im_parse = torch.from_numpy(parse_array) #[0,19]
        
        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        shape = self.transform(parse_shape) # [-1,1]
        phead = torch.from_numpy(parse_head) # [0,1]
        pcm = torch.from_numpy(parse_cloth) # [0,1]

        # upper cloth
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
        pcm = pcm.unsqueeze(0)
        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

        # load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform(im_pose)
        
        # cloth-agnostic representation
        agnostic = torch.cat([shape, im_h, pose_map], 0) 

        if self.stage == 'GMM':
            im_g = Image.open(self.opt.grid_path)
            im_g = self.transform(im_g)
        else:
            im_g = ''

        if self.stage == 'GMM':
            result = {
                'c_name':   c_name,     # for visualization
                'im_name':  im_name,    # for visualization or ground truth
                'cloth':    c,          # for input
                'image':    im,         # for visualization
                'agnostic': agnostic,   # for input
                'parse_cloth': im_c,    # for ground truth
                'head': im_h,           # for visualization
                'grid_image': im_g,     # for visualization
                }

        if self.stage == 'SEG':
            result = {
                'c_name':   c_name,     # for visualization
                'im_name':  im_name,    # for visualization or ground truth
                'cloth':    c,          # for input
                'image':    im,         # for visualization
                'agnostic': agnostic,   # for input
                'parse_model':im_parse, # for ground truth
                }
        if self.stage == 'TOM':
            result = {
                'c_name':   c_name,     # for visualization
                'im_name':  im_name,    # for visualization or ground truth
                'cloth':    c,          # for input
                'image':    im,         # for visualization
                'agnostic': agnostic,   # for input
                'parse_cloth_mask':pcm, # for ground truth
                'texture_t_prior': im_ttp, # for input
                'parse_cloth': im_c,    # for ground truth
                'head': im_h,           # for visualization
                }


        return result

    def __len__(self):
        return len(self.im_names)
    
    
    
class SieveDataLoader(object):
    def __init__(self, opt, dataset):
        super(SieveDataLoader, self).__init__()

        self.runmode = opt.runmode
        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            if self.runmode == 'train':
                self.data_iter = self.data_loader.__iter__()
                batch = self.data_iter.__next__()
            if self.runmode == 'test' :
                batch = None
        return batch
   