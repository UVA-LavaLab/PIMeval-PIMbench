import argparse
import torch
import torch.nn as nn
import time

# Function to perform layer normalization
def perform_layer_norm(input_tensor, norm_layer, device):
    input_tensor = input_tensor.to(device)
    norm_layer = norm_layer.to(device)

    start_time = time.time()
    output = norm_layer(input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Wait for GPU ops to complete
    end_time = time.time()

    elapsed_time = end_time - start_time
    return elapsed_time

# Main function
def main(args):
    # Set device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Input tensor: [B, C, H, W]
    input_tensor = torch.randn(args.batch_size, args.input_channels, args.input_height, args.input_width)

    # LayerNorm normalized over [C, H, W] for each sample
    normalized_shape = [args.input_channels, args.input_height, args.input_width]
    norm_layer = nn.LayerNorm(normalized_shape, eps=args.epsilon)

    # Run layer normalization
    time_taken = perform_layer_norm(input_tensor, norm_layer, device)
    print(f"[INFO] Time taken for layer normalization: {time_taken * 1000:.6f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN-style Layer Normalization on CPU/GPU")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-c", "--input_channels", type=int, default=64, help="Number of input channels")
    parser.add_argument("-r", "--input_height", type=int, default=32, help="Input height")
    parser.add_argument("-w", "--input_width", type=int, default=32, help="Input width")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-5, help="Epsilon for LayerNorm")
    parser.add_argument("-cuda", "--cuda", action='store_true', help="Use CUDA if available")

    args = parser.parse_args()
    main(args)
