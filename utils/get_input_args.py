#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/final_project/train.py
#
# PROGRAMMER: Lucio Yovanny Nogales Vera
# DATE CREATED: 7/09/2023
# REVISED DATE:
# PURPOSE: Create and train the classifier using pre-trained Neural Networks
##

# Imports python modules
import argparse

def get_input_args_to_train():
    """
    Retrieves and parses the 8 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to created and defined these 8 command line arguments. If
    the user fails to provide some or all of the 8 arguments, then the default
    values are used for the missing arguments.
    Command Line Arguments:
      1. Path to images to predict as data_dir with default value 'flowers/'
      2. Pre-trained model architecture to use as --arch with default value 'vgg16'
      3. Path to load model checkpoint file as --save-dir with default value 'checkpoints/'
      4. Number of classes to compare and plot as --top_classes_k with default value '5'
      5. Learning rate for the optimizer as --learning_rate with default value '0.01'
      6. Number of hidden units for the classifier as --hidden_units with default value '512'
      7. Number of epochs for the training as --epochs with default value '2'
      8. Flag to activate the GPU use for training as --gpu with default value 'False'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('data_dir', type = str, default = 'flowers/',
                        help = 'path to the folder of images to predict')
    parser.add_argument('--arch', type = str, default = 'vgg16',
                        help = 'pre-trained model architecture to use')
    parser.add_argument('--save-dir', type = str, default = 'checkpoints/',
                        help = 'path to load model checkpoint file')
    parser.add_argument('--learning_rate', type = float, default = 0.01,
                        help = 'learning rate for the optimizer')
    parser.add_argument('--hidden_units', type = int, default = 4096,
                        help = 'number of hidden units for the classifier')
    parser.add_argument('--epochs', type = int, default = 2,
                        help = 'number of epochs for the training')
    parser.add_argument('--gpu', type = bool, default = False,
                        help = 'flag to activate the GPU use for training')

    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    return parser.parse_args()

def get_input_args_to_predict():
    """
    Retrieves and parses the 4 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to created and defined these 4 command line arguments. If
    the user fails to provide some or all of the 4 arguments, then the default
    values are used for the missing arguments.
    Command Line Arguments:
      1. Path to image to predict as image_path
      2. Path to load model checkpoint file as checkpoint with default value 'checkpoint'
      3. Category names file to load as category_names with default value 'cat_to_name.json'
      4. Flag to activate the GPU use for training as
      5. Number of top classes to compare and plot as --top_k with default value '5'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
        parse_args() -data structure that stores the command line arguments object
        """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('image_path', type = str,
                        help = 'path to the folder of image to predict')
    parser.add_argument('checkpoint', type = str,
                        help = 'checkpoint file to load')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                        help = 'category names file to load')
    parser.add_argument('--gpu', type = bool, default = False,
                        help = 'flag to activate the GPU use for training')
    parser.add_argument('--top_k', type = int, default = 3,
                        help = 'number of top classes to compare and plot')

    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    return parser.parse_args()