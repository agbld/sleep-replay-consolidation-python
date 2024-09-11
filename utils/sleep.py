from utils.nn import SimpleNN
import torch
import numpy as np
import warnings

def create_masked_input(X, numexamples, mask_size):
    warnings.warn("create_masked_input is deprecated and will be removed in a future version.", DeprecationWarning)
    sleep_input = np.mean(X, axis=0)
    sleep_input = sleep_input.reshape(28, 28)
    sleep_x = np.zeros((numexamples, 28, 28))

    for i in range(numexamples):
        x_pos = np.random.randint(0, 28 - mask_size)
        y_pos = np.random.randint(0, 28 - mask_size)
        sleep_x[i, x_pos:x_pos + mask_size, y_pos:y_pos + mask_size] = sleep_input[x_pos:x_pos + mask_size, y_pos:y_pos + mask_size]

    sleep_x = sleep_x.reshape(numexamples, 784)
    return sleep_x

def normalize_nn_data(nn: SimpleNN, x):

    with torch.no_grad():
        factor_log = []
        
        # Forward propagate the input data
        nn.eval()  # Set the network to evaluation mode (disabling dropout)
        activations = nn.forward(torch.Tensor(x).to(device))  # Forward propagate through the network
        nn.train()  # Set back to training mode after forward pass
        
        previous_factor = 1.0
        
        # Iterate over each layer (assuming nn.W is a list of weight matrices and activations)
        for l in range(len(nn.layers)):
            # Get the maximum weight and maximum activation
            weight_max = np.max(np.maximum(0, nn.layers[l].weight.data.cpu().numpy()))
            activation_max = np.max(np.maximum(0, activations[l+1].cpu().numpy()))  # activations[l+1] is the next layer's activation
            
            # Calculate the scaling factor
            scale_factor = max(weight_max, activation_max)
            applied_inv_factor = scale_factor / previous_factor
            
            # Rescale the weights
            nn.layers[l].weight.data /= applied_inv_factor
            
            # Store the factor log
            factor_log.append(1.0 / applied_inv_factor)
            
            # Update previous factor for the next layer
            previous_factor = applied_inv_factor

    return nn, factor_log