# Inferencing using a pre-trained VGG13 on CPU or GPU with CUDA support
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
import time
import argparse

# Command-line argument parsing
parser = argparse.ArgumentParser(description='VGG13 Inference')
parser.add_argument('-s', '--device', required=True, choices=['cpu', 'cuda'], help='Device to use for inference (cpu/cuda)')
parser.add_argument('-n', '--num_images', type=int, default=1000, help='Number of images to test')
args = parser.parse_args()

deviceStr = args.device
if deviceStr != 'cpu' and deviceStr != 'cuda':
    print("[ERROR] Invalid device! Must be either cpu or cuda.")
    exit(-1)
if deviceStr == 'cuda' and not torch.cuda.is_available():
    print("[WARNING] cuda is not available. Falling back to cpu.")
    deviceStr = 'cpu'
print("[INFO] Using " + deviceStr)
device = torch.device(deviceStr)

def data_loader(data_dir, batch_size, num_images, shuffle=True, test=False):
    print("[INFO] Starting data loader")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
    ])

    if test:
        dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
        dataset = Subset(dataset, range(num_images))
    else:
        dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True if device == 'cuda' else False
    )

    print("[INFO] Data loader finished")
    return data_loader

num_classes = 100  # CIFAR-100 has 100 classes
if deviceStr == 'cpu':
    batch_size = 128  # Adjust this based on your CPU's memory capacity
if deviceStr == 'cuda':
    batch_size = 2  # Adjust this based on your GPU's memory capacity

print("[INFO] Loading test data")
test_loader = data_loader(data_dir='./data', batch_size=batch_size, num_images=args.num_images, test=True)
print("[INFO] Test data loaded")

# Load pre-trained VGG13 model
print("[INFO] Loading pre-trained VGG13 model")
model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
print("[INFO] Pre-trained VGG13 model loaded")

# Modify the classifier to match the number of classes
# classifier[6] corresponds to the last dense layer
model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(device)

# Optimize the model with TorchScript
model = torch.jit.script(model)

criterion = nn.CrossEntropyLoss()

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

    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total

    print(f'Accuracy of the network on the test images (Top-1): {top1_accuracy:.2f} %')
    print(f'Accuracy of the network on the test images (Top-5): {top5_accuracy:.2f} %')
    print(f"Number of images: {args.num_images}")
    print(f"Execution time per image: {(tm/args.num_images):.4f} seconds")
