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
from torchvision import models
import torch
from torch import nn, optim
from utils.get_input_args import get_input_args_to_train
from utils.check_command_line_arguments import check_command_line_arguments_to_train
from utils.load_data_loaders import load_data_loaders
from utils.create_model import create_model
from utils.save_checkpoint import save_checkpoint

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

def main():
    in_arg = get_input_args_to_train()
    check_command_line_arguments_to_train(in_arg)
    image_datasets, dataloaders = load_data_loaders(data_dir, train_dir, valid_dir, test_dir)

    out_units = len(image_datasets['train'].classes)

    model = create_model(in_arg.arch, in_arg.hidden_units, out_units)

    model, optimizer = train_model(model, dataloaders, in_arg.epochs, in_arg.gpu, in_arg.learning_rate)

    save_checkpoint(model, in_arg.save_dir, optimizer, in_arg.arch, in_arg.hidden_units, out_units, in_arg.epochs, in_arg.learning_rate, in_arg.gpu, out_units)

def train_model(model, dataloaders, epochs, use_gpu, learning_rate):
    print("Training model...")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    counter = 0
    epochs = epochs
    by_batches = 15
    device = torch.device("cuda" if use_gpu else "cpu")

    model.to(device)

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_train_loss = 0
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            counter += 1
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            running_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            if counter % by_batches == 0:
                valid_loss = 0
                test_correct = 0  # Number of correct predictions on the test set

                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    for images, labels in dataloaders['test']:
                        images, labels = images.to(device), labels.to(device)

                        output = model.forward(images)
                        loss = criterion(output, labels)
                        valid_loss += loss.item()

                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        test_correct += equals.sum().item()

                # Get mean loss to enable comparison between train and test sets
                train_loss = running_train_loss / by_batches
                test_loss = valid_loss / len(dataloaders['valid'].dataset)

                # At completion of epoch
                train_losses.append(train_loss)
                test_losses.append(test_loss)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(train_loss),
                      "Test Loss: {:.3f}.. ".format(test_loss),
                      "Test Accuracy: {:.3f}".format(test_correct / len(dataloaders['test'].dataset)))

    print("Model trained!")

    return model, optimizer

# Call to main function to run the program
if __name__ == "__main__":
    main()