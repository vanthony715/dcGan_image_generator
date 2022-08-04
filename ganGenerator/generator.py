#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:08:37 2021

@author: avasquez
"""

import os
import sys
import gc
import time
import utils

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms

##clear bins
gc.collect()
torch.cuda.empty_cache()

IMG_CHANNEL = 3
G_HID = 64
X_DIM = 64
D_HID = 64
BATCH_SZ = 1
D_HIDDEN = 64
Z_DIM = 100
NumSyntheticSamples = 8505

CUDA = True
CUDA = CUDA and torch.cuda.is_available()
print('Pytorch Version: {}'.format(torch.__version__))
if CUDA:
    print('CUDA version: {}'.format(torch.version.cuda))
cudnn.benchmark = True
device = torch.device("cuda:0" if CUDA else "cpu")

##generator network
class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self.main = nn.Sequential(
            ##layer 1
            nn.ConvTranspose2d(Z_DIM, G_HID * 8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(G_HID * 8), nn.ReLU(True),
            ##layer 2
            nn.ConvTranspose2d(G_HID * 8, G_HID * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(G_HID * 4), nn.ReLU(True),
            ##layer 3
            nn.ConvTranspose2d(G_HID * 4, G_HID * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(G_HID * 2), nn.ReLU(True),
            ##layer 4
            nn.ConvTranspose2d(G_HID * 2, G_HID , 4, 2, 1, bias = False),
            nn.BatchNorm2d(G_HID), nn.ReLU(True),
            ##output layer
            nn.ConvTranspose2d(G_HID, IMG_CHANNEL, 4, 2, 1, bias = False),
            nn.Tanh())
    def forward(self, input):
        return self.main(input)

if __name__ == "__main__":
    ##start clock
    tic = time.time()
    
    ##define paths
    mnt = '/mnt/opsdata/neurocondor/datasets/avasquez/data/'
    basepath = mnt + 'GAN/chip96x96/'
    weightsPath = basepath + 'trucks-augOutput/netG_29.pth'
    writePath = basepath + 'gan/'
    
    
    weights = weightsPath
    ##run net on data
    gNet = GNet().to(device)
    print(gNet)
    gNet.load_state_dict(torch.load(weights))
    for j in range(1, NumSyntheticSamples):
        name = 'semi_3'+ '_' + str(j).zfill(5)
        syntheticSample = utils.syntheticGen(gNet, device, BATCH_SZ, Z_DIM)
        utils.saveImage(writePath, syntheticSample, name)

    gc.collect()
    ##Clock time
    print('\n----Time----\n')
    toc = time.time()
    tf = round((toc - tic), 1)
    print('Time to Run (s): ', tf)