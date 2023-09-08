#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/final_project/utils/create_model.py
#
# PROGRAMMER: Lucio Yovanny Nogales Vera
# DATE CREATED: 7/09/2023
# REVISED DATE:
# PURPOSE: Create model using pre-trained Neural Networks
##
from torch import nn
from torchvision import models
from collections import OrderedDict

def create_model(arch, hidden_units, out_units):
    model = None

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.2, inplace=False)),
        ('fc2', nn.Linear(hidden_units, int(hidden_units/2))),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.5, inplace=False)),
        ('fc3', nn.Linear(int(hidden_units/2), out_units)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    print('Model created successfully!')

    return model
