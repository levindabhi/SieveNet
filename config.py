
import os.path as osp
import os


class parser:
    def __init__(self):
        self.dataroot = 'dataset'
        self.datamode = 'train'                                   #train, test
        self.stage = 'TOM'                                        #GMM, SEG, TOM
        self.runmode = self.datamode                              
        self.name = self.stage   
        if self.datamode == 'train':
            self.data_list = 'train_pairs.txt'
        elif self.datamode == 'test':
            self.data_list = 'test_pairs.txt'
        self.fine_width = 192
        self.fine_height = 256
        self.radius = 4
        self.grid_path =  osp.join(self.dataroot, 'grid.png')
        if self.datamode == 'train':                            #for training keep true, for test keep false
            self.shuffle = True             
        else:
            self.shuffle = False
        self.batch_size = 16
        self.workers = 1
        self.grid_size = 5
        
        self.lr = 0.002
        self.keep_step = 8000
        self.decay_step = 5500
        self.previous_step = 0                                  #if you want to resume training from some steps    
                                                                #set previous_step in as per last updated checkpoints 
        self.save_count = 200
        self.display_count = 50
        
        self.tensorboard_dir = osp.join(os.getcwd(), 'tensorboard')
        self.checkpoint_dir = osp.join(os.getcwd(), 'checkpoints')
        self.save_dir = osp.join(os.getcwd(), 'outputs')         #for saving output while infernce
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.previous_step == 0:
            self.checkpoint = ''
        else:
            self.checkpoint = osp.join(self.checkpoint_dir, self.name, 'step_%06d.pth' % (self.previous_step))

        self.input_image_path = 'custom/input/019579_0.jpg'
        self.cloth_image_path = 'custom/input/017575_1.jpg'
        self.human_parsing_image_path = 'custom/input/019579_0.png'