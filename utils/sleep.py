from utils.nn import SimpleNN
import torch
import numpy as np

def create_masked_input(X, numexamples, mask_size):
    sleep_input = np.mean(X, axis=0)
    sleep_input = sleep_input.reshape(28, 28)
    sleep_x = np.zeros((numexamples, 28, 28))

    for i in range(numexamples):
        x_pos = np.random.randint(0, 28 - mask_size)
        y_pos = np.random.randint(0, 28 - mask_size)
        sleep_x[i, x_pos:x_pos + mask_size, y_pos:y_pos + mask_size] = sleep_input[x_pos:x_pos + mask_size, y_pos:y_pos + mask_size]

    sleep_x = sleep_x.reshape(numexamples, 784)
    return sleep_x

def get_activations(nn: SimpleNN, x, dim=None):
    # Hook function to capture the activations
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Assuming you have a PyTorch model called `model`
    activations = {}

    # Registering hook to each layer (this example assumes sequential model)
    for name, layer in nn.layers.named_modules():
        layer.register_forward_hook(get_activation(name))

    # Forward pass through the model
    _ = nn(x)

    return activations

def get_spikes(nn: SimpleNN, x, threshold, dim=None):
    # Get the activations
    activations = get_activations(nn, x)

    # Calculate the spikes for each layer
    spikes = {}
    for name, activation in activations.items():
        spikes[name] = (activation > threshold).float()