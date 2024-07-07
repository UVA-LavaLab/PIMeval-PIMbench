import torch
import torchvision.models as models
import numpy as np
import csv

# Load pre-trained VGG13 model
vgg13 = models.vgg13(pretrained=True)

# Set the model to evaluation mode
vgg13.eval()

# Function to binarize weights
def binarize_weights(weights):ak
    #return (weights > 0).astype(int)
    return weights

# Save the VGG13 weights to a CSV file
def save_weights_to_csv(model, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for name, param in model.named_parameters():
            if 'weight' in name:  # We only want the weights, not biases
                weights = param.detach().numpy()
                binarized_weights = binarize_weights(weights)
                shape = binarized_weights.shape
                # Flatten and save the binarized weights
                writer.writerow([name] + list(shape) + binarized_weights.flatten().tolist())

# Save the VGG13 weights to a CSV file
save_weights_to_csv(vgg13, 'vgg13_weights.csv')


