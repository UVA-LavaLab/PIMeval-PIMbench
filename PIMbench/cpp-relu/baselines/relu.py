import argparse
import torch
import torch.nn as nn
import time

# Function to perform ReLU operation
def perform_relu(input_tensor, relu_layer, device):
    input_tensor = input_tensor.to(device)
    relu_layer = relu_layer.to(device)
    
    start_time = time.time()
    output = relu_layer(input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Wait for all GPU operations to complete
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    return elapsed_time

# Main function
def main(args):
    # Set the device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Initialize input tensor
    input_tensor = torch.randn(args.batch_size, args.input_depth, args.input_height, args.input_width)

    # Define ReLU layer
    relu_layer = nn.ReLU()

    # Perform the ReLU operation and measure time
    time_taken = perform_relu(input_tensor, relu_layer, device)
    print(f"[INFO] Time taken for ReLU for batch size of {args.batch_size}: {time_taken * 1000:.6f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform ReLU on CPU or CUDA")
    
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for the input")
    parser.add_argument("-d", "--input_depth", type=int, default=128, help="Depth (channels) of the input")
    parser.add_argument("-r", "--input_height", type=int, default=226, help="Height of the input")
    parser.add_argument("-c", "--input_width", type=int, default=226, help="Width of the input")
    parser.add_argument("-cuda", "--cuda", action='store_true', help="Use GPU if available")
    
    args = parser.parse_args()
    main(args)
