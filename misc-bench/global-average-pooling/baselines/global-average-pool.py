import argparse
import torch
import torch.nn as nn
import time

# Function to perform global average pooling operation
def perform_global_average_pooling(input_tensor, pool_layer, device):
    input_tensor = input_tensor.to(device)
    pool_layer = pool_layer.to(device)
    
    start_time = time.time()
    output = pool_layer(input_tensor)
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

    # Define global average pooling layer
    pool_layer = nn.AdaptiveAvgPool2d((1, 1))

    # Perform the global average pooling and measure time
    time_taken = perform_global_average_pooling(input_tensor, pool_layer, device)
    print(f"[INFO] Time taken for global average pooling for batch size of {args.batch_size}: {time_taken * 1000:.6f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform global average pooling on CPU or CUDA")
    
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for the input")
    parser.add_argument("-d", "--input_depth", type=int, default=64, help="Depth (channels) of the input")
    parser.add_argument("-r", "--input_height", type=int, default=224, help="Height of the input")
    parser.add_argument("-c", "--input_width", type=int, default=224, help="Width of the input")
    parser.add_argument("-cuda", "--cuda", action='store_true', help="Use GPU if available")
    
    args = parser.parse_args()
    main(args)

