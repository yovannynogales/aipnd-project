#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/final_project/utils/check_command_line_arguments.py
#
# PROGRAMMER: Lucio Yovanny Nogales Vera
# DATE CREATED: 7/09/2023
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE:  Print the command line arguments as check for argparse
#
##
import glob

def check_command_line_arguments_to_train(in_arg):
    """
    Prints each of the command line arguments passed in as parameter in_arg,
    assumes you defined all three command line arguments.
    Parameters:
     in_arg -data structure that stores the command line arguments object
    Returns:
     Nothing - just prints to console
    """
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args_to_train' hasn't been defined.")
    else:
        if in_arg.arch not in ['vgg16', 'densenet121', 'vgg19']:
            raise Exception("The architecture is not valid. Please choose between 'vgg16', 'densenet121' or 'vgg19'")

        if len(glob.glob(in_arg.data_dir)) == 0:
            raise Exception("The data directory is not valid. Please choose a valid directory")

        if in_arg.epochs < 1:
            raise Exception("The epoch number is not valid. Please choose a valid number")

        # prints command line agrs
        print("Command Line Arguments:\n     data_dir", in_arg.data_dir,
              "\n     save_dir", in_arg.save_dir, "\n     arch", in_arg.arch,
              "\n     learning_rate", in_arg.learning_rate, "\n     hidden_units", in_arg.hidden_units,
              "\n     epochs", in_arg.epochs, "\n     gpu", in_arg.gpu
             )

def check_command_line_arguments_to_predict(in_arg):
    """
    Prints each of the command line arguments passed in as parameter in_arg,
    assumes you defined all three command line arguments.
    Parameters:
     in_arg -data structure that stores the command line arguments object
    Returns:
     Nothing - just prints to console
    """
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args_to_train' hasn't been defined.")
    else:
        if len(glob.glob(in_arg.image_path)) == 0:
            raise Exception("The image path is not valid. Please choose a valid path")

        if len(glob.glob(in_arg.checkpoint)) < 1:
            raise Exception("The checkpoint path is not valid. Please choose a valid path")

        if len(glob.glob(in_arg.category_names)) < 1:
            raise Exception("The category names path is not valid. Please choose a valid path")

        # prints command line agrs
        print("Command Line Arguments:\n     image path", in_arg.image_path,
              "\n     checkpoint", in_arg.checkpoint, "\n     category_names", in_arg.category_names,
              "\n     gpu", in_arg.gpu, "\n     top_k", in_arg.top_k
             )