################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

import tqdm

from cifar100_utils import get_train_validation_set, get_test_set, add_augmentation

def make_noisy(image, device):
    # takes a batch of images and applies noise accoring \
    # normal distribution
    z = torch.randn(image.size()).to(device)
    noise = 0.1 * z
    noisy_image= image + noise 

    return noisy_image
  


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model =models.resnet18(weights='DEFAULT')

    # Randomly initialize and modify the model's last layer for CIFAR100.
    for param in model.parameters():
        param.requires_grad = False
    
    std= 0.1
    model.fc = nn.Linear(512, num_classes)
    model.fc.bias.data.fill_(0)
    model.fc.weight.data.normal_(0, std=std)
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True 


    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    train_dataset, validation_dataset = get_train_validation_set(data_dir, validation_size=5000, augmentation_name=None)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    
    # Initialize the optimizer (Adam) to train the last layer of the model.
    top_accuracy = 0
    # Training loop with validation after each epoch. Save the best model.
    for epoch in range(epochs):
        model.train()
        
        for data in tqdm.tqdm(train_loader):
            x, target = data 
            x, target = x.to(device), target.to(device)
            #x = make_noisy(x)
            optimizer.zero_grad()
            y = model(x)
                
            loss = criterion(y, target)
            loss.backward()
            optimizer.step()
        val_acc = evaluate_model(model, val_loader, device)

        if val_acc > top_accuracy:
            top_accuracy =  val_acc
            top_model = model
            torch.save({
            'best_acc1': top_accuracy,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_name)
        


    # Load the best model on val accuracy and return it.
    model = top_model

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """

    model.to(device)
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()
    N = 0
    test_acc = 0
    # Loop over the dataset and compute the accuracy. Return the accuracy
    for data in data_loader:
        x, labels = data
        x, labels = x.to(device), labels.to(device)
        #x = make_noisy(x, device)
        N += labels.size(0)
        with torch.no_grad():
            y_pred = model(x)
        y_pred  = torch.argmax(y_pred, dim = 1 )
        test_acc += torch.sum(y_pred == labels)
        
    accuracy = test_acc/N


    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

    # Load the model
    model = get_model()
    model = model.to(device)
    # Set model to eval mode
    model.eval()


    # Get the augmentation to use
    transform_list = []
    add_augmentation(augmentation_name, transform_list)

    # Train the model
    checkpoint_name = './model'
    model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name)
    # Evaluate the model on the test set
    test_loader = get_test_set(data_dir)
    val_loader = data.DataLoader(dataset = test_loader, batch_size=batch_size, shuffle=False, drop_last=False)

    accuracy = evaluate_model(model, val_loader, device)
    print(accuracy)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
