#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/final_project/predict.py
#
# PROGRAMMER: Lucio Yovanny Nogales Vera
# DATE CREATED: 7/09/2023
# REVISED DATE:
# PURPOSE: Predict the flower name from an image with predict.py along with the probability of that name using an already trained deep learning model.
##

# Imports python modules
import torch
from utils.get_input_args import get_input_args_to_predict
from utils.check_command_line_arguments import check_command_line_arguments_to_predict
from utils.load_checkpoint import load_checkpoint
from utils.load_category_names import load_category_names
from utils.process_image import process_image

def main():
    in_arg = get_input_args_to_predict()
    check_command_line_arguments_to_predict(in_arg)
    model, optimizer = load_checkpoint(in_arg.checkpoint)
    cat_to_name = load_category_names(in_arg.category_names)
    probs, classes = predict(in_arg.image_path, model, in_arg.top_k, in_arg.gpu)

    print('Predict program finished successfully!')
    for i in range(len(classes)):
        print(f'Flower name: {cat_to_name[classes[i]]} with probability: {probs[i]}')

def predict(image_path, model, topk, use_gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda" if use_gpu else "cpu")

    model.to(device)
    model.eval()

    tensor = torch.from_numpy(process_image(image_path)).to(device, dtype=torch.float)
    print(tensor.shape)

    output = model.forward(tensor.unsqueeze_(0))

    probabilities = torch.exp(output)

    top_probabilities, top_classes = probabilities.topk(topk, dim=1)
    top_probabilities, top_classes = top_probabilities.cpu(), top_classes.cpu()

    class_to_idx_inverse = {model.class_to_idx[k]: k for k in model.class_to_idx}

    mapped_labels = []
    for label in top_classes.detach().numpy()[0]:
        mapped_labels.append(class_to_idx_inverse[label])

    return top_probabilities.detach().numpy()[0], mapped_labels


# Call to main function to run the program
if __name__ == "__main__":
    main()