#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/final_project/utils/load_checkpoint.py
#
# PROGRAMMER: Lucio Yovanny Nogales Vera
# DATE CREATED: 7/09/2023
# REVISED DATE:
# PURPOSE: Load the checkpoint and rebuild the model
##
import torch
from torch import optim
from utils.create_model import create_model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model_arch = checkpoint['model']
    model_hidden_units = checkpoint.get('hidden_units', 4096)
    model_out_units = checkpoint.get('out_units', 102)

    model = create_model(model_arch, model_hidden_units, model_out_units)

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer_for_model = optimizer.load_state_dict(state_dict=checkpoint['optimizer'])

    print('Model loaded successfully!')

    return model, optimizer_for_model