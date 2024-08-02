# Inference using a pre-trained VGG16 model on CPU or GPU with CUDA support
import argparse
import os
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load and preprocess a batch of images from a given list of file paths
def load_images(image_paths, batch_size):
    print(f"[INFO] Loading images from: {image_paths}")
    
    preprocess = transforms.Compose([
        transforms.Resize(256),           # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),       # Crop the center 224x224 pixels of the image
        transforms.ToTensor(),            # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image tensor
    ])
    
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = preprocess(image)
        images.append(image)

    # Handle case where batch size is greater than the number of test images
    while len(images) < batch_size:
        images += images[:batch_size - len(images)]
    
    images = torch.stack(images[:batch_size])  # Stack images to create a batch (B, C, H, W)
    return images

# Get the appropriate device (CPU or GPU) based on user preference and availability
def get_device(use_cuda):
    print("[INFO] Determining computation device")
    if use_cuda and not torch.cuda.is_available():
        print("[WARNING] cuda is not available. Falling back to cpu.")
        return torch.device("cpu") 
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Perform inference on a batch of images using the specified model and device
def predict(images, model, device):
    print("[INFO] Performing inference")
    model.eval()  # Set the model to evaluation mode
    images = images.to(device)  # Move the image tensor to the appropriate device
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(images)  # Perform inference
    # Apply softmax to get class probabilities        
    probabilities = torch.nn.functional.softmax(output, dim=1)
    return probabilities

# Load category names from a file into a list
def load_categories(file_path):
    print(f"[INFO] Loading categories from: {file_path}")
    categories = [None] * 1000  # Assuming 1000 categories as VGG16 is trained on ImageNet dataset
    with open(file_path, 'r') as file:
        for line in file:
            index, category = line.strip().split(": ")  # Split line into index and category
            index = int(index)  # Convert index to integer
            category = category.strip("'")  # Remove extra quotes around category
            categories[index] = category  # Assign category to the correct index
    return categories

# Process all images in a given directory and perform classification
def process_directory(directory, model, categories, device, batch_size):
    print(f"[INFO] Processing images in directory: {directory}")
    print(f"[INFO] Batch size: {batch_size}");
    total_time = 0  # Total execution time
    total_images = 0  # Total number of images processed

    # Iterate through all files in the directory    
    image_paths = []
    for image_name in os.listdir(directory):
        image_path = os.path.join(directory, image_name)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(image_path)
    
    # Warm-up iterations to allow the GPU to complete any initialization steps (data movement) and optimize its execution pipeline
    if device.type == 'cuda':
        for i in range(0, len(image_paths[:batch_size]), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images = load_images(batch_paths, batch_size)
            _ = predict(images, model, device)
        torch.cuda.synchronize()

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = load_images(batch_paths, batch_size)
        
        start_time = time.time()
        probabilities = predict(images, model, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()  # Wait for all GPU operations to complete
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        total_time += elapsed_time  # Record the time taken for inference
        total_images += len(batch_paths)

        # Get the top 5 predictions        
        for j in range(len(batch_paths)):
            print(f"Results for image {os.path.basename(batch_paths[j])}:")
            top5_prob, top5_catid = torch.topk(probabilities[j], 5)
            for k in range(top5_prob.size(0)):
                print(f"  {categories[top5_catid[k]]}: {top5_prob[k].item():.4f}")
            print()

    avg_time_per_image = total_time / total_images if total_images else 0
    print(f"Average execution time per image: {avg_time_per_image * 1000:.4f} ms")

# Main function to handle command line arguments, load the model, and process images
def main(args):
    print("[INFO] Starting main function")
    # Load the categories from the provided file
    categories = load_categories(args.categories)

    # Get the appropriate device (CPU or GPU)
    device = get_device(args.cuda)
    print(f"[INFO] Using device: {device}")

    # Load the pre-trained VGG16 model
    print("[INFO] Loading VGG16 model")
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)

    # Process all images in the specified directory in batches
    batch_size = args.cuda_batch_size if device.type == 'cuda' else args.cpu_batch_size
    process_directory(args.directory, model, categories, device, batch_size=batch_size)

if __name__ == "__main__":
    print("[INFO] Parsing command line arguments")
    parser = argparse.ArgumentParser(description="Image classification using VGG16")
    parser.add_argument("-d", "--directory", type=str, default="../../cpp-vgg13/baselines/data/test/", help="Path to the directory containing images")
    parser.add_argument("-c", "--categories", type=str, default="../../cpp-vgg13/baselines/categories.txt", help="Path to the categories text file")
    parser.add_argument("-cuda", action='store_true', help="Use GPU if available")
    parser.add_argument("-cuda_batch_size", type=int, default=64, help="Batch size for inference on GPU")
    parser.add_argument("-cpu_batch_size", type=int, default=64, help="Batch size for inference on CPU")
    args = parser.parse_args()
    print("[INFO] Starting the process")
    main(args)
