import argparse
import torch
import torch.nn as nn
import time

# Function to perform batch normalization operation
def perform_batch_normalization(input_tensor, bn_layer, device):
    input_tensor = input_tensor.to(device)
    bn_layer = bn_layer.to(device)
    
    start_time = time.time()
    output = bn_layer(input_tensor)
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
    input_tensor = torch.randn(args.batch_size, args.num_features, args.height, args.width)

    # Define batch normalization layer
    bn_layer = nn.BatchNorm2d(num_features=args.num_features, 
                              eps=args.eps, 
                              momentum=args.momentum,
                              affine=args.affine,
                              track_running_stats=args.track_running_stats)

    # Perform the batch normalization and measure time
    time_taken = perform_batch_normalization(input_tensor, bn_layer, device)
    print(f"[INFO] Time taken for batch normalization for batch size of {args.batch_size}: {time_taken * 1000:.6f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform batch normalization on CPU or CUDA")
    
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for the input")
    parser.add_argument("-f", "--num_features", type=int, default=64, help="Number of features (channels)")
    parser.add_argument("-r", "--height", type=int, default=224, help="Height of the input")
    parser.add_argument("-c", "--width", type=int, default=224, help="Width of the input")
    parser.add_argument("-e", "--eps", type=float, default=1e-5, help="Epsilon value for numerical stability")
    parser.add_argument("-m", "--momentum", type=float, default=0.1, help="Momentum for running mean and variance")
    parser.add_argument("-a", "--affine", action='store_true', default=True, help="Use affine transformation (learnable parameters)")
    parser.add_argument("-t", "--track_running_stats", action='store_true', default=True, help="Track running statistics")
    parser.add_argument("-cuda", "--cuda", action='store_true', help="Use GPU if available")
    
    args = parser.parse_args()
    main(args) 