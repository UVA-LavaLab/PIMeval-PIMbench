# Inference using a pre-trained MLP model on CPU or GPU with CUDA support
import sys
import time
import argparse
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Define fully connected layers
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        
        # Combine all layers into a sequential module
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Determine the device dynamically
        device = next(self.parameters()).device

        # Move input to the device
        x = x.to(device)

        # Forward pass through the MLP
        return self.mlp(x)

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

# Parse the '-l' arguement into a list of ints that represent the number of neurons at each layer.
# The first element is the number of input neurons, the last is the number out output neurons, and the hidden neurons are in between.
#   Default execution example: "500,1000,2" -> [500, 100, 2]
def parse_layer_config_arg():
    layer_string_arg = args.l
    res = [int(i) for i in layer_string_arg.split(',')]
    if len(res) < 3:
        print("[ERROR] not enough layers specified, must specify at least three layers. \nexiting...")
        sys.exit(1)
    return res

def run_mlp(device):
    layers = parse_layer_config_arg()  # get a list of neurons for each layer into a list

    input_size = layers[0]  # Size of input features
    hidden_sizes = layers[1:-1]  # Sizes of hidden layers
    output_size = layers[-1]  # Number of output neurons

    # Create an MLP and move it to the device
    mlp = MLP(input_size, hidden_sizes, output_size).to(device)

    # Generate random input data
    input_data = torch.randn(args.n, input_size).to(device)

    # Record the start time
    start_time = time.time()

    # Forward pass
    output = mlp(input_data)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time in milliseconds
    elapsed_time_ms = (end_time - start_time) * 1000

    print("Elapsed time for forward pass: {:.2f} milliseconds".format(elapsed_time_ms))

# Main function to handle command line arguments and load the model
def main(args):
    print("[INFO] Starting main function")

    # Get the appropriate device (CPU or GPU)
    device = get_device(args.cuda)
    print(f"[INFO] Using device: {device}")

    run_mlp(device)


if __name__ == "__main__":
    print("[INFO] Parsing command line arguments")
    parser = argparse.ArgumentParser(description="runs an MLP inference, where the layers are defined by the user")
    parser.add_argument("-cuda", action='store_true', help="Use GPU if available")
    parser.add_argument("-l", default="500,1000,2", help="comma-seperate list of number the neurons in each layer (default=\"500,1000,2\")")
    parser.add_argument("-n", type=int, default=1, help="number of inference points for the batch (default=1)")
    args = parser.parse_args()
    print("[INFO] Starting the inference process")
    main(args)
