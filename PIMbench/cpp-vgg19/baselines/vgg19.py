# Inferencing using a pre-trained VGG19 on CPU or GPU with CUDA support
import argparse
import os
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load and preprocess an image from a given file path
def load_image(image_path):

    print(f"[INFO] Loading image from: {image_path}")
    # Open the image file
    image = Image.open(image_path)
    
    # Define preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),           # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),       # Crop the center 224x224 pixels of the image
        transforms.ToTensor(),            # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image tensor
    ])
    
    # Apply the preprocessing transformations and add a batch dimension
    image = preprocess(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
    return image

# Get the appropriate device (CPU or GPU) based on user preference and availability
def get_device(use_cuda):

    print("[INFO] Determining computation device")
    # Check if GPU is available and if the user wants to use it
    if use_cuda and not torch.cuda.is_available():
        print("[WARNING] cuda is not available. Falling back to cpu.")
        return torch.device("cpu") 
    if use_cuda and torch.cuda.is_available():
        print("[INFO] Using GPU")
        return torch.device("cuda")
    else:
        print("[INFO] Using CPU")
        return torch.device("cpu")

# Perform inference on a given image using the specified model and device
def predict(image, model, device):

    print("[INFO] Performing inference")
    model.eval()  # Set the model to evaluation mode
    image = image.to(device)  # Move the image tensor to the appropriate device
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(image)  # Perform inference
    # Apply softmax to get class probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

# Load category names from a file into a list
def load_categories(file_path):

    print(f"[INFO] Loading categories from: {file_path}")
    categories = [None] * 1000  # Assuming 1000 categories as VGG19 is trained on ImageNet dataset
    with open(file_path, 'r') as file:
        for line in file:
            index, category = line.strip().split(": ")  # Split line into index and category
            index = int(index)  # Convert index to integer
            category = category.strip("'")  # Remove extra quotes around category
            categories[index] = category  # Assign category to the correct index
    return categories

# Process all images in a given directory and perform classification
def process_directory(directory, model, categories, device):

    print(f"[INFO] Processing images in directory: {directory}")
    times = []  # List to store the execution times for each image
    
    # Iterate through all files in the directory
    for image_name in os.listdir(directory):
        image_path = os.path.join(directory, image_name)  # Full path to the image file
        
        # Skip files that are not images
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"[INFO] Skipping non-image file: {image_name}")
            continue

        print(f"[INFO] Processing image: {image_name}")
        # Load and preprocess the image
        image = load_image(image_path)
        
        # Measure execution time for inference
        start_time = time.time()
        probabilities = predict(image, model, device)
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)  # Record the time taken for inference
        
        # Get the top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        print(f"Results for image {image_name}:")
        for i in range(top5_prob.size(0)):
            print(f"  {categories[top5_catid[i]]}: {top5_prob[i].item():.4f}")
        print()

    # Calculate and print the average execution time per image
    avg_time = sum(times) / len(times) if times else 0
    print(f"Average execution time per image: {avg_time * 1000:.4f} ms")

# Main function to handle command line arguments, load the model, and process images
def main(args):

    print("[INFO] Starting main function")
    # Load the categories from the provided file
    categories = load_categories(args.categories)

    # Get the appropriate device (CPU or GPU)
    device = get_device(args.cuda == 't')
    print(f"[INFO] Using device: {device}")

    # Load the pre-trained VGG19 model
    print("[INFO] Loading VGG19 model")
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)

    # Process all images in the specified directory
    process_directory(args.directory, model, categories, device)

if __name__ == "__main__":
    # Parse command line arguments
    print("[INFO] Parsing command line arguments")
    parser = argparse.ArgumentParser(description="Image classification using VGG19")
    parser.add_argument("-d", "--directory", type=str, default="../../cpp-vgg13/baselines/data/test/", help="Path to the directory containing images")
    parser.add_argument("-c", "--categories", type=str, default="../../cpp-vgg13/baselines/categories.txt", help="Path to the categories text file")
    parser.add_argument("-cuda", type=str, choices=['t', 'f'], default='f', help="Use GPU if available ('t' for true, 'f' for false)")
    args = parser.parse_args()
    print("[INFO] Starting the process")
    main(args)
