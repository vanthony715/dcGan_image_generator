#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:23:39 2021

@author: avasquez
"""
import os
import shutil
import cv2
import torch
from tqdm import tqdm
import torchvision.utils as vutils
import xml.etree.ElementTree as ET

def clearFolder(Path):
    if os.path.isdir(Path):
        print('Removing File: ', Path)
        shutil.rmtree(Path)
    print('Creating File: ', Path)
    os.mkdir(Path)

def syntheticGen(Network, Device, BATCH_SZ, Z_DIM):
    visNoise = torch.randn(BATCH_SZ, Z_DIM, 1, 1, device = Device)
    with torch.no_grad():
        syntheticSample = Network(visNoise)
    return syntheticSample

def saveImage(OutPath, Sample, Name):
    SavePath = OutPath + Name  + '.jpg'
    vutils.save_image(Sample, SavePath, normalize = True)