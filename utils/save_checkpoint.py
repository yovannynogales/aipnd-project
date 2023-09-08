#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/final_project/utils/save_checkpoint.py
#
# PROGRAMMER: Lucio Yovanny Nogales Vera
# DATE CREATED: 7/09/2023
# REVISED DATE:
# PURPOSE: Save checkpoint
##
import torch

def save_checkpoint(model, save_dir, optimizer, arch, hidden_units, out_units, epochs, learning_rate, use_gpu, class_to_idx):
    checkpoint = {'model': arch,
                  'hidden_units': hidden_units,
                  'out_units': out_units,
                  'epoch': epochs,
                  'learning_rate': learning_rate,
                  'use_gpu': use_gpu,
                  'class_to_idx': class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, f'{save_dir}/checkpoint-{arch}.pth')
    print('Checkpoint saved!')
