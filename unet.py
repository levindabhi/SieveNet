import torch
import torch.nn as nn

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#baseline of this sievenet i.e., cp-viton is using up-sampling and then conv operation
#while original u-net paper cited in sievenet is using upconvolution
#architecture of this net is avaliable in VITON paper https://arxiv.org/abs/1711.08447

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=True, use_bias=True):
        super(UnetGenerator, self).__init__()

        if use_dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None
        self.uprelu = nn.ReLU(True)
        self.downrelu = nn.LeakyReLU(0.2, True)

        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
        
        self.conv2 = nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.norm2 = nn.BatchNorm2d(ngf*2)
        
        self.conv3 = nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.norm3 = nn.BatchNorm2d(ngf*4)
        
        self.conv4 = nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.norm4 = nn.BatchNorm2d(ngf*8)
        
        self.conv5 = nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.norm5 = nn.BatchNorm2d(ngf*8)
        
        self.conv6 = nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.norm6 = nn.BatchNorm2d(ngf*8)

        self.upconv6 = nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm6 = nn.BatchNorm2d(ngf*8)

        #here input channel is doubled because of soft connection
        self.upconv5 = nn.ConvTranspose2d(ngf*8*2, ngf*8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm5 = nn.BatchNorm2d(ngf*8)
        
        self.upconv4 = nn.ConvTranspose2d(ngf*8*2, ngf*4, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm4 = nn.BatchNorm2d(ngf*4)

        self.upconv3 = nn.ConvTranspose2d(ngf*4*2, ngf*2, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm3 = nn.BatchNorm2d(ngf*2)

        self.upconv2 = nn.ConvTranspose2d(ngf*2*2, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm2 = nn.BatchNorm2d(ngf)
        
        self.upconv1 = nn.ConvTranspose2d(ngf*1*2, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

    def forward(self, input):

        x1 = self.conv1(input)
        x2 = self.operation(x1, self.downrelu, self.conv2, self.norm2)
        x3 = self.operation(x2, self.downrelu, self.conv3, self.norm3)
        x4 = self.operation(x3, self.downrelu, self.conv4, self.norm4)
        x5 = self.operation(x4, self.downrelu, self.conv5, self.norm5)
        x6 = self.operation(x5, self.downrelu, self.conv6, self.norm6)

        x = self.operation(x6, self.uprelu, self.upconv6, self.upnorm6, dropout=self.dropout)
        x = self.operation(torch.cat([x, x5], dim=1), self.uprelu, self.upconv5, self.upnorm5)
        x = self.operation(torch.cat([x, x4], dim=1), self.uprelu, self.upconv4, self.upnorm4)
        x = self.operation(torch.cat([x, x3], dim=1), self.uprelu, self.upconv3, self.upnorm3)
        x = self.operation(torch.cat([x, x2], dim=1), self.uprelu, self.upconv2, self.upnorm2)
        
        x = self.uprelu(torch.cat([x,x1], dim=1))
        x = self.upconv1(x)

        return(x)

    def operation(self, input, activation, conv, norm, dropout=None):
        x = activation(input)
        x = conv(x)
        x = norm(x)
        if dropout is not None:
            x = dropout(x)
        return x