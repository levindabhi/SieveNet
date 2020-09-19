# SieveNet #
This is the unofficial implementation of 'SieveNet: A Unified Framework for Robust Image-Based Virtual Try-On' </br>
Paper can be found from [here](https://arxiv.org/pdf/2001.06265.pdf)

# Dataset downloading and processing #
Dataset download instructions and link of dataset can be found from official repo of [CP-VTON](https://github.com/sergeywong/cp-vton) and [VITON](https://github.com/xthan/VITON) </br>
Put dataset in `data` folder

# Usage #
Clone the repo and install requirements through ```requirement.txt``` 

# Traning #
#### Coarse-to-Fine Warping module ####
&nbsp;&nbsp;&nbsp;&nbsp; In `config.py` set ```self.datamode='train'``` and ```self.stage='GMM'```
</br> &nbsp;&nbsp;&nbsp;&nbsp; then run ```python train.py```
</br> You can observe results while traning in tensorboard as below
</br>
![SS from tensorboard while training gmm](https://github.com/levindabhi/SieveNet/image/train_gmm.png)

####  Conditional Segmentation Mask generation module ####
&nbsp;&nbsp;&nbsp;&nbsp; In `config.py` set ```self.datamode='Train'``` and ```self.stage='SEG'```
</br> &nbsp;&nbsp;&nbsp;&nbsp; then run ```python train.py```
</br>
![SS from tensorboard while training segm](https://github.com/levindabhi/SieveNet/image/train_segm.jpeg)

####  Segmentation Assisted Texture Translation module ####
&nbsp;&nbsp;&nbsp;&nbsp; In `config.py` set ```self.datamode='Train'``` and ```self.stage='TOM'```
</br> &nbsp;&nbsp;&nbsp;&nbsp; then run ```python train.py```
</br>
![SS from tensorboard while training tom](https://github.com/levindabhi/SieveNet/image/train_tom.png)


# Testing on dataset #
Please download checkpoint of all three modules from [google drive](www.google.com) and put them in `checkpoints` folder
</br>
For testing, in `config.py` set ```self.datamode='test'```
</br> For Testing of Coarse-to-Fine Warping module, Conditional Segmentation Mask generation module, and Segmentation Assisted Texture Translation module set ```self.stage='GMM'```, ```self.stage='SEG'```, and ```self.stage='TOM'``` respectively.
</br>
Here is testing result. For Coarse-to-Fine Warping module,
</br>
![SS from tensorboard while testing gmm](https://github.com/levindabhi/SieveNet/image/test_gmm.png)
</br>For Segmentation Assisted Texture Translation module, 
</br>
![SS from tensorboard while testing gmm](https://github.com/levindabhi/SieveNet/image/test_tom.png)

# Testing on custom image #
1. Please download checkpoint of all three modules from [google drive](www.google.com) and put them in `checkpoints` folder.
2. Please download caffe-model from [here](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel) and put the model in `pose` folder. </br>
3. Generate human parsing from [Self-Correction-Human-Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing) repo. Select `LIP` dataset while generating human parsing.</br>
4. Set input-image's, cloth-image's, and output of human parsing image's path in config file.</br>
5. Then run ```python inference.py```
Output will be saved in `outputs` folder.

# Acknowledgements #
Some modules of this implementation is based on this [repo](https://github.com/sergeywong/cp-vton)</br>
For generating pose keypoints, I have used [learnopencv](https://github.com/spmallick/learnopencv/tree/master/OpenPose-Multi-Person) implementation of OpenPose
