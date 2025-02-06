# PyTorch code to run convolution on CPU or GPU with CUDA support
import argparse
import torch
import torch.nn as nn
import time

# Function to perform convolution operation
def perform_convolution(input_tensor, conv_layer, device):
    input_tensor = input_tensor.to(device)
    conv_layer = conv_layer.to(device)
    
    start_time = time.time()
    output = conv_layer(input_tensor)
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

    # Define convolution layer
    conv_layer = nn.Conv2d(in_channels=args.input_depth, out_channels=args.kernel_depth, 
                           kernel_size=(args.kernel_height, args.kernel_width),
                           stride=args.stride, padding=args.padding)

    # Perform the convolution and measure time
    time_taken = perform_convolution(input_tensor, conv_layer, device)
    print(f"[INFO] Time taken for convolution for batch size of {args.batch_size}: {time_taken * 1000:.6f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform convolution on CPU or CUDA")
    
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for the input")
    parser.add_argument("-d", "--input_depth", type=int, default=64, help="Depth (channels) of the input")
    parser.add_argument("-r", "--input_height", type=int, default=224, help="Height of the input")
    parser.add_argument("-c", "--input_width", type=int, default=224, help="Width of the input")
    parser.add_argument("-kr", "--kernel_height", type=int, default=3, help="Height of the convolution kernel")
    parser.add_argument("-kc", "--kernel_width", type=int, default=3, help="Width of the convolution kernel")
    parser.add_argument("-kd", "--kernel_depth", type=int, default=64, help="Number of output channels (depth) of the convolution")
    parser.add_argument("-s", "--stride", type=int, default=1, help="Stride of the convolution")
    parser.add_argument("-p", "--padding", type=int, default=1, help="Padding for the convolution")
    parser.add_argument("-cuda", "--cuda", action='store_true', help="Use GPU if available")
    
    args = parser.parse_args()
    main(args)
