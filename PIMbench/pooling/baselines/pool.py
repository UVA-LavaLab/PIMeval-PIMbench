import argparse
import torch
import torch.nn as nn
import time

# Function to perform max pooling operation
def perform_max_pooling(input_tensor, pool_layer, device):
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

    # Define max pooling layer
    pool_layer = nn.MaxPool2d(kernel_size=(args.kernel_height, args.kernel_width),
                              stride=args.stride, padding=args.padding)

    # Perform the max pooling and measure time
    time_taken = perform_max_pooling(input_tensor, pool_layer, device)
    print(f"[INFO] Time taken for max pooling for batch size of {args.batch_size}: {time_taken * 1000:.6f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform max pooling on CPU or CUDA")
    
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for the input")
    parser.add_argument("-d", "--input_depth", type=int, default=64, help="Depth (channels) of the input")
    parser.add_argument("-r", "--input_height", type=int, default=224, help="Height of the input")
    parser.add_argument("-c", "--input_width", type=int, default=224, help="Width of the input")
    parser.add_argument("-kh", "--kernel_height", type=int, default=2, help="Height of the pooling kernel")
    parser.add_argument("-kw", "--kernel_width", type=int, default=2, help="Width of the pooling kernel")
    parser.add_argument("-s", "--stride", type=int, default=2, help="Stride of the pooling")
    parser.add_argument("-p", "--padding", type=int, default=0, help="Padding for the pooling")
    parser.add_argument("-cuda", "--cuda", action='store_true', help="Use GPU if available")
    
    args = parser.parse_args()
    main(args)

