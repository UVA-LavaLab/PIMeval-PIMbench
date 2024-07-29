# Inferencing using a pre-trained VGG19 on CPU or GPU with CUDA support
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
import time
import argparse
import os

# Command-line argument parsing for specifying device and number of images
parser = argparse.ArgumentParser(description='VGG19 Inference')
parser.add_argument('-s', '--device', required=True, choices=['cpu', 'cuda'], help='Device to use for inference (cpu/cuda)')
parser.add_argument('-n', '--num_images', type=int, default=1000, help='Number of images to test')
args = parser.parse_args()

# Determine the device to use for inference
deviceStr = args.device
if deviceStr != 'cpu' and deviceStr != 'cuda':
    print("[ERROR] Invalid device! Must be either cpu or cuda.")
    exit(-1)
if deviceStr == 'cuda' and not torch.cuda.is_available():
    print("[WARNING] cuda is not available. Falling back to cpu.")
    deviceStr = 'cpu'
print("[INFO] Using " + deviceStr)
device = torch.device(deviceStr)

# Function to load data
def data_loader(data_dir, batch_size, num_images, shuffle=True, test=False):
    print("[INFO] Starting data loader")

    # Normalization parameters for the images    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Transformation to apply to the images
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
    ])

    # Load the test dataset
    dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    dataset = Subset(dataset, range(num_images))

    # num_workers specifies the number of subprocesses used for data loading. 
    # Each subprocess independently loads and preprocesses data, which can then be fed to the GPU for training or inference.
    num_workers = os.cpu_count() # This will create a subprocess for each CPU core
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device == 'cuda' else False
    )

    print("[INFO] Data loader finished")
    return data_loader

num_classes = 100  # CIFAR-100 has 100 classes

# Adjust the batch_size below based on the CPU or GPU's memory capacity.
# Memory capacity refers to the amount of memory (RAM) available on the CPU or GPU, typically measured in gigabytes (GB).
# The batch size needs to be adjusted so that it fits within the system's memory. If the batch size is too large, it can result in out-of-memory (OOM) errors.
if deviceStr == 'cpu':
    batch_size = 128 
if deviceStr == 'cuda':
    batch_size = 2 

# Load the test data
print("[INFO] Loading test data")
test_loader = data_loader(data_dir='./data', batch_size=batch_size, num_images=args.num_images, test=True)
print("[INFO] Test data loaded")

# Load pre-trained VGG19 model
print("[INFO] Loading pre-trained VGG19 model")
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
print("[INFO] Pre-trained VGG19 model loaded")

# Modify the classifier to match the number of classes
# classifier[6] corresponds to the last dense layer
model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(device)

# Optimize the model with TorchScript
model = torch.jit.script(model)

model.eval() # Set model to evaluation mode
with torch.no_grad():
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    tm = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        start = time.time()
        outputs = model(images)
        _, predicted_top1 = torch.max(outputs.data, 1)
        _, predicted_top5 = torch.topk(outputs, 5, dim=1)
        end = time.time()
        tm += end - start
        total += labels.size(0)
        correct_top1 += (predicted_top1 == labels).sum().item()
        correct_top5 += sum([1 if labels[i] in predicted_top5[i] else 0 for i in range(len(labels))])

    # Calculate accuracy
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total

    # Print the results
    print(f'Accuracy of the network on the test images (Top-1): {top1_accuracy:.2f} %')
    print(f'Accuracy of the network on the test images (Top-5): {top5_accuracy:.2f} %')
    print(f"Number of images: {args.num_images}")
    print(f"Execution time per image: {(tm / args.num_images * 1000):.4f} ms")
