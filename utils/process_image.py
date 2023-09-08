#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/final_project/utils/process_image.py
#
# PROGRAMMER: Lucio Yovanny Nogales Vera
# DATE CREATED: 7/09/2023
# REVISED DATE:
# PURPOSE: Process an image path into a PyTorch tensor
##
import numpy as np
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_img = Image.open(image).convert('RGB')

    image_size = pil_img.size
    h = min(image_size)
    w = max(image_size)

    ratio_aspec = w/h

    x = image_size.index(min(image_size))
    y = image_size.index(max(image_size))

    new_image_size = [0, 0]
    new_image_size[x] = 256
    new_image_size[y] = int(new_image_size[x] * ratio_aspec)
    pil_img = pil_img.resize(new_image_size)

    width, height = new_image_size

    left_margin = (width - 224)/2
    top_margin = (height - 224)/2
    right_margin = (width + 224)/2
    bottom_margin = (height + 224)/2

    pil_img = pil_img.crop((left_margin, top_margin, right_margin, bottom_margin))

    np_image = np.array(pil_img)

    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std

    np_image = np_image.transpose((2, 0, 1))

    return np_image