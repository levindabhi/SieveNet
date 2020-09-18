# SieveNet #
This is the unofficial implementation of 'SieveNet: A Unified Framework for Robust Image-Based Virtual Try-On' </br>
Paper can be found from [here](https://arxiv.org/pdf/2001.06265.pdf)

# Dataset downloading and processing #
Dataset download instructions and link of dataset can be found from official repo of [CP-VTON](https://github.com/sergeywong/cp-vton) and [VITON](https://github.com/xthan/VITON) </br>
Put dataset in `data` folder

# Traning #
To install requirements please run `requirement.txt`
#### Coarse-to-Fine Warping module ####
&nbsp;&nbsp;&nbsp;&nbsp; In `config.py` set ```self.datamode='train'``` and ```self.stage='GMM'```
</br> &nbsp;&nbsp;&nbsp;&nbsp; then run ```python train.py```

####  Conditional Segmentation Mask generation module ####
&nbsp;&nbsp;&nbsp;&nbsp; In `config.py` set ```self.datamode='Train'``` and ```self.stage='SEG'```
</br> &nbsp;&nbsp;&nbsp;&nbsp; then run ```python train.py```

####  Segmentation Assisted Texture Translation module ####
&nbsp;&nbsp;&nbsp;&nbsp; In `config.py` set ```self.datamode='Train'``` and ```self.stage='TOM'```
</br> &nbsp;&nbsp;&nbsp;&nbsp; then run ```python train.py```

# Testing on dataset #
Please download checkpoint of all three modules from [google drive](www.google.com) and put them in `checkpoints` folder
</br>
For testing, in `config.py` set ```self.datamode='test'```
</br> For Testing of Coarse-to-Fine Warping module, Conditional Segmentation Mask generation module, and Segmentation Assisted Texture Translation module set ```self.stage='GMM'```, ```self.stage='SEG'```, and ```self.stage='TOM'``` respectively.
