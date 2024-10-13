from typing import Tuple
from utils.nn import SimpleNN
import torch
import numpy as np
import warnings
import copy
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sleep_phase(nn: SimpleNN, num_iterations: int, sleep_opts: dict, X: torch.Tensor, device: torch.device = device,
                callback_func = None, callback_steps = 0, acc_df = None,
                save_best = False) -> Tuple[SimpleNN, dict]:
    nn = copy.deepcopy(nn)  # Create a deep copy of the network to avoid modifying the original network

    nn_best = None
    acc_all_best = 0

    nn.eval()  # Set the network to evaluation mode (disabling dropout)

    # [Differences from the original] 
    # 1. Using a slightly different method for mask creation.
    # 2. Using the mean of sampled X for each iteration instead of the entire X.
    # Generate sleep input based on the mean of the input samples and a random mask.
    sleep_input = np.zeros((num_iterations, X.shape[1]))
    for i in range(num_iterations):
        mask = np.random.choice([0, 1], size=X.shape[1], p=[1 - sleep_opts['mask_fraction'], sleep_opts['mask_fraction']])
        sampled_X_for_iter = X[np.random.choice(X.shape[0], sleep_opts['samples_per_iter'])]
        mean_sample = np.mean(sampled_X_for_iter, axis=0)
        sleep_input[i] = mean_sample * mask
    sleep_input = torch.Tensor(sleep_input).to(device)

    nn_size = [nn.layers[0].in_features] + [layer.out_features for layer in nn.layers]

    membrane_potentials = [torch.zeros(size).to(device) for size in nn_size]
    spikes = [torch.zeros(size).to(device) for size in nn_size]
    refrac_end = [torch.zeros(size).to(device) for size in nn_size]

    with torch.no_grad():
        with tqdm(total=num_iterations) as pbar:
            
            # Initialize log variables
            accum_dW_inc = [torch.zeros_like(layer.weight) for layer in nn.layers]
            accum_dW_dec = [torch.zeros_like(layer.weight) for layer in nn.layers]
            accum_H_inc = [torch.zeros_like(layer.weight) for layer in nn.layers]
            accum_H_dec = [torch.zeros_like(layer.weight) for layer in nn.layers]
            last_accum_H_inc = [torch.zeros_like(layer.weight) for layer in nn.layers]
            last_accum_H_dec = [torch.zeros_like(layer.weight) for layer in nn.layers]

            for t in range(num_iterations):
                # Create Poisson-distributed spikes from the input images
                # The input is randomly spiked at each time step
                rescale_factor = 1 / (sleep_opts['dt'] * sleep_opts['max_rate']) # Rescale factor based on maximum firing rate
                spike_snapshots = torch.rand(nn_size[0]).to(device) * rescale_factor / 2 # Generate random spikes
                input_spikes = (spike_snapshots <= sleep_input[t]).float() # Compare to sleep input to determine spikes

                # Fisrt layer spikes are based on the input directly
                spikes[0] = torch.Tensor(input_spikes).to(device)

                # Update the membrane potentials and spikes for each layer (starting from second layer)
                for l in range(1, len(nn_size)):

                    # Compute the input impulse based on spikes from the previous layer, update membrane potentials
                    impulse = nn.layers[l - 1](spikes[l - 1]) * sleep_opts['alpha'][l - 1]
                    impulse = impulse - torch.mean(impulse) * sleep_opts['W_inh'] # Apply inhibition
                    membrane_potentials[l] = membrane_potentials[l] * sleep_opts['decay'] + impulse

                    # Add a direct current (DC) component for layer 4, if needed
                    if l == len(nn_size) - 1:
                        membrane_potentials[l] += sleep_opts['DC']
                    
                    # Update spiking state based on membrane potential and threshold
                    threshold = sleep_opts['threshold'] * sleep_opts['beta'][l - 1]
                    spikes[l] = torch.Tensor((membrane_potentials[l] >= threshold).float())

                    # Get pre-synaptic spikes and post-synaptic spikes
                    pre = spikes[l - 1].unsqueeze(0)  # (num_pre_neurons,) -> (1, num_pre_neurons)
                    post = spikes[l].unsqueeze(1)  # (num_post_neurons,) -> (num_post_neurons, 1)
        
                    # Spike-Timing Dependent Plasticity (STDP) to adjust weights
                    sigmoid_weights = torch.sigmoid(nn.layers[l - 1].weight)

                    dW_inc = sleep_opts['inc'] * (post == 1) * (pre == 1) * sigmoid_weights
                    dW_dec = sleep_opts['dec'] * (post == 1) * (pre == 0) * sigmoid_weights
                    dW = dW_inc - dW_dec

                    # Update log variables
                    accum_dW_inc[l - 1] += dW_inc
                    accum_dW_dec[l - 1] += dW_dec
                    accum_H_inc[l - 1] += (post == 1) * (pre == 1)
                    accum_H_dec[l - 1] += (post == 1) * (pre == 0)

                    # Update weights
                    nn.layers[l - 1].weight += dW

                    # Reset the membrane potential of spiking neurons
                    membrane_potentials[l][spikes[l] == 1] = 0

                    # Update the refractory period for spiking neurons
                    refrac_end[l][spikes[l] == 1] = t + sleep_opts['t_ref']

                if (callback_func is not None) and (t % callback_steps == 0 or t == num_iterations - 1):
                    # calculate the l2 norm of the dW_inc, dW_dec
                    dW_inc_norm = [torch.norm(torch.flatten(dW), p=2).item() for dW in accum_dW_inc]
                    dW_dec_norm = [torch.norm(torch.flatten(dW), p=2).item() for dW in accum_dW_dec]

                    # calculate the l2 norm of the H_inc, H_dec
                    H_inc_norm = [torch.norm(torch.flatten(H_inc), p=2).item() for H_inc in accum_H_inc]
                    H_dec_norm = [torch.norm(torch.flatten(H_dec), p=2).item() for H_dec in accum_H_dec]

                    # calculate the l2 norm of the dH_inc, dH_dec
                    dH_inc = [accum_H_inc[i] - last_accum_H_inc[i] for i in range(len(accum_H_inc))]
                    dH_dec = [accum_H_dec[i] - last_accum_H_dec[i] for i in range(len(accum_H_dec))]
                    dH_inc_norm = [torch.norm(torch.flatten(dH), p=2).item() for dH in dH_inc]
                    dH_dec_norm = [torch.norm(torch.flatten(dH), p=2).item() for dH in dH_dec]

                    # Reset the log variables
                    accum_dW_inc = [torch.zeros_like(layer.weight) for layer in nn.layers]
                    accum_dW_dec = [torch.zeros_like(layer.weight) for layer in nn.layers]
                    last_accum_H_inc = copy.deepcopy(accum_H_inc)
                    last_accum_H_dec = copy.deepcopy(accum_H_dec)
                    accum_H_inc = [torch.zeros_like(layer.weight) for layer in nn.layers]
                    accum_H_dec = [torch.zeros_like(layer.weight) for layer in nn.layers]

                    args = {
                        'steps': t,
                        'dW_inc_norm': dW_inc_norm,
                        'dW_dec_norm': dW_dec_norm,
                        'dH_inc_norm': dH_inc_norm,
                        'dH_dec_norm': dH_dec_norm,
                        'H_inc_norm': H_inc_norm,
                        'H_dec_norm': H_dec_norm
                    }

                    acc_df = callback_func(nn, acc_df, args)
                    if save_best:
                        acc_dict = acc_df[-1]
                        if acc_dict['All'] > acc_all_best:
                            acc_all_best = acc_dict['All']
                            nn_best = copy.deepcopy(nn)

                pbar.update(1)

            # avg_spikes = [total / num_iterations for total in total_spikes]
            # print(avg_spikes)

            # Normalize weights if required by sleep_opts
            if sleep_opts['normW'] == 1:
                for l in range(1, len(nn_size)):
                    nn.layers[l - 1].weight = sleep_opts['gamma'] * nn.layers[l - 1].weight / (torch.max(nn.layers[l - 1].weight) - torch.min(nn.layers[l - 1].weight)) * \
                        (torch.max(sleep_opts['W_old'][l - 1]) - torch.min(sleep_opts['W_old'][l - 1]))
                
    if save_best:
        return nn_best, acc_df
    return nn, acc_df

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