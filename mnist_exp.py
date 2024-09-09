#%%
# Import necessary libraries
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils.nn import SimpleNN
from utils.nn import train_network, evaluate_all, evaluate_per_task, log_accuracy
from utils.task import create_class_task
from utils.sleep import create_masked_input, get_activations, get_spikes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# MNIST Dataset Loading

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Convert datasets to DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Convert to numpy format for task generation (optional)
train_x = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255
train_y = train_dataset.targets.numpy()
test_x = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255
test_y = test_dataset.targets.numpy()

#%%
# Task Generation

num_tasks = 5
task_size = (10 + num_tasks - 1) // num_tasks
print(f'Number of Tasks: {num_tasks}')
print(f'Task Size: {task_size}')
train_tasks = create_class_task(train_y, task_size)
test_tasks = create_class_task(test_y, task_size)

sequential_train_x = []

for task_id in range(num_tasks):
    task_indices = np.where(train_tasks == task_id)[0]
    task_train_x = train_x[task_indices[:1000]]  # Use the first 5000 samples for each task
    sequential_train_x.append(task_train_x)

sequential_train_x = np.concatenate(sequential_train_x)

#%%
# Configuration

acc_df = []

opts = {
    'numepochs': 2,         # Number of epochs
    'batchsize': 100,       # Batch size for training
    'learning_rate': 0.01,   # Learning rate for SGD
    'momentum': 0.5         # Momentum for SGD
}

nn_size_template = [784, 4000, 4000, 10]

#%%
# Exp 1: Sleep Replay Consolidation (SRC)

def sleep_phase(nn: SimpleNN, num_iterations: int, sleep_opts: dict, sleep_input: torch.Tensor):
    nn.eval()  # Set the network to evaluation mode (disabling dropout)

    sleep_input = sleep_input.to(device)

    nn_size = [nn.layers[0].in_features] + [layer.out_features for layer in nn.layers]

    membrane_potentials = [torch.zeros(size).to(device) for size in nn_size]
    spikes = [torch.zeros(size).to(device) for size in nn_size]
    refrac_end = [torch.zeros(size).to(device) for size in nn_size]

    with torch.no_grad():
        with tqdm(total=num_iterations) as pbar:
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

                    # Spike-Timing Dependent Plasticity (STDP) to adjust weights
                    def stdp(weight, pre, post):
                        # Compute the weight delta using broadcasting
                        sigmoid_weights = torch.sigmoid(weight)

                        # Compute the weight delta using broadcasting
                        weight_inc = sleep_opts['inc'] * (post == 1) * (pre == 1) * sigmoid_weights
                        weight_dec = sleep_opts['dec'] * (post == 1) * (pre == 0) * sigmoid_weights

                        # Combine increments and decrements
                        return weight_inc - weight_dec

                    # Get pre-synaptic spikes and post-synaptic spikes
                    pre = spikes[l - 1].unsqueeze(0)  # (num_pre_neurons,) -> (1, num_pre_neurons)
                    post = spikes[l].unsqueeze(1)  # (num_post_neurons,) -> (num_post_neurons, 1)
                    
                    # Compute the weight delta
                    weight_delta = stdp(nn.layers[l - 1].weight, pre, post)

                    # Update weights
                    nn.layers[l - 1].weight += weight_delta

                    # Reset the membrane potential of spiking neurons
                    membrane_potentials[l][spikes[l] == 1] = 0

                    # Update the refractory period for spiking neurons
                    refrac_end[l][spikes[l] == 1] = t + sleep_opts['t_ref']

                pbar.update(1)

            # avg_spikes = [total / num_iterations for total in total_spikes]
            # print(avg_spikes)

            # Normalize weights if required by sleep_opts
            if sleep_opts['normW'] == 1:
                for l in range(1, len(nn_size)):
                    nn.layers[l - 1].weight = sleep_opts['gamma'] * nn.layers[l - 1].weight / (torch.max(nn.layers[l - 1].weight) - torch.min(nn.layers[l - 1].weight)) * \
                        (torch.max(sleep_opts['W_old'][l - 1]) - torch.min(sleep_opts['W_old'][l - 1]))
                
    return nn

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

def run_sleep_exp(acc_df: list, sleep_opts_update={}):

    # src_model = SimpleNN([784, 1200, 1200, 10])
    src_model = SimpleNN(nn_size_template).to(device)

    # Define the hyperparameters for the sleep phase
    sleep_opts = {
        'iterations': 1,
        'beta': [14.548273, 44.560317, 38.046326],
        'alpha_scale': 55.882454,
        'alpha': [14.983829, 253.17746, 7.7707720],
        'decay': 0.999,
        'W_inh': 0.0,
        'inc': 0.032064,
        'dec': 0.003344,
        'max_rate': 239.515363,
        'threshold': 1.0,
        'DC': 0.0,
        'normW': 0,
        'gamma': 1.0,
        't_ref': 0,
        'dt': 0.001,
    }

    sleep_opts.update(sleep_opts_update)
    print(sleep_opts)

    acc_df = log_accuracy(f'SRC', 'Initial', acc_df, src_model, test_x, test_y, test_tasks, sleep_opts)

    for task_id in range(num_tasks):
        print(f'Task {task_id}')

        task_indices = np.where(train_tasks == task_id)[0]
        task_train_x = train_x[task_indices[:5000]]  # Use the first 5000 samples for each task
        task_train_y = train_y[task_indices[:5000]]
        
        src_model = train_network(src_model, task_train_x, task_train_y, opts)

        # [Visual] Record activations before SRC for layer visualization
        with torch.no_grad():
            src_model.eval()
            activations_before = get_activations(src_model.to(device), torch.Tensor(sequential_train_x).to(device))
            layer_activations_before = [act.cpu() for act in activations_before.values()]

        print('Before SRC: ', evaluate_per_task(src_model, test_x, test_y, test_tasks, num_tasks))
        
        acc_df = log_accuracy(f'SRC', 'Task ' + str(task_id) + ' Before SRC', acc_df, src_model, test_x, test_y, test_tasks, sleep_opts)

        # Generate masked input for the sleep phase
        sleep_period = int(sleep_opts['iterations'] + task_id * float(sleep_opts['iterations']) / 3)
        sleep_input = create_masked_input(task_train_x, sleep_period, 10)

        # Calculate the alpha
        # _, factor_log = normalize_nn_data(src_model, task_train_x)
        # sleep_opts['alpha'] = [alpha * sleep_opts['alpha_scale'] for alpha in factor_log]
        # print('alpha: ', sleep_opts['alpha'])

        # Run the sleep phase
        src_model = sleep_phase(src_model, sleep_period, sleep_opts, torch.tensor(sleep_input))
        
        print('After SRC: ', evaluate_per_task(src_model, test_x, test_y, test_tasks, num_tasks))

        acc_df = log_accuracy(f'SRC', 'Task ' + str(task_id) + ' After SRC', acc_df, src_model, test_x, test_y, test_tasks, sleep_opts)

        # [Visual] Record activations after SRC for layer visualization
        with torch.no_grad():
            src_model.eval()

            activations_after = get_activations(src_model.to(device), torch.Tensor(sequential_train_x).to(device))
            layer_activations_after = [act.cpu() for act in activations_after.values()]

        # [Visual] Fit PCA to the activations and reduce the dimensionality
        reduced_activations_before = []
        reduced_activations_after = []
        reduced_activations_diff = []
        for i in range(len(src_model.layers)):
            if i == len(src_model.layers) - 1:
                reduced_activations_before.append(layer_activations_before[i])
                reduced_activations_after.append(layer_activations_after[i])

            pca = PCA(n_components=10)
            layer_activations = np.concatenate((layer_activations_before[i], layer_activations_after[i]))
            reduced_activations = pca.fit_transform(layer_activations)

            reduced_activations_before.append(reduced_activations[:len(layer_activations_before[i])])
            reduced_activations_after.append(reduced_activations[len(layer_activations_before[i]):])
            reduced_activations_diff.append(reduced_activations_after[i] - reduced_activations_before[i])

        # [Visual] Plot the reduced activations as images for each layer before and after SRC
        fig, axs = plt.subplots(3, len(src_model.layers), figsize=(len(src_model.layers) * 6, 12))
        fig.suptitle(f'Layer Activations for Task {task_id}: Before, After SRC, and Difference', fontsize=18)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

        for i in range(len(src_model.layers)):
            im1 = axs[0, i].imshow(reduced_activations_before[i], aspect='auto')
            axs[0, i].set_title(f'Layer {i} - Before SRC', fontsize=12)
            im2 = axs[1, i].imshow(reduced_activations_after[i], aspect='auto')
            axs[1, i].set_title(f'Layer {i} - After SRC', fontsize=12)
            im3 = axs[2, i].imshow(reduced_activations_diff[i], aspect='auto')
            axs[2, i].set_title(f'Layer {i} - Diff.', fontsize=12)

        fig.colorbar(im3, cax=cbar_ax)

        fig.savefig(f'./png/layer_activations_task_{task_id}_before_after_src.png')

    acc_df = log_accuracy(f'SRC', 'After Training', acc_df, src_model, test_x, test_y, test_tasks, sleep_opts)

    print(evaluate_all(src_model, test_x, test_y))

    return acc_df

for iteration in [400]:
    acc_df = run_sleep_exp(
        acc_df, 
        {
            'iterations': iteration, 
            'inc': 0.001,
            'dec': 0.0001,
        },)

#%%
# Exp 2: Sequential Training
control_model = SimpleNN(nn_size_template).to(device)

acc_df = log_accuracy('Sequential', 'Initial', acc_df, control_model, test_x, test_y, test_tasks)

for task_id in range(num_tasks):
    task_indices = np.where(train_tasks == task_id)[0]
    task_train_x = train_x[task_indices[:5000]]  # Train on the first 5000 samples for each task
    task_train_y = train_y[task_indices[:5000]]
    
    control_model = train_network(control_model, task_train_x, task_train_y, opts)

    print(evaluate_per_task(control_model, test_x, test_y, test_tasks, num_tasks))

    acc_df = log_accuracy('Sequential', 'Task ' + str(task_id), acc_df, control_model, test_x, test_y, test_tasks)

acc_df = log_accuracy('Sequential', 'After Training', acc_df, control_model, test_x, test_y, test_tasks)

print(evaluate_all(control_model, test_x, test_y))

#%%
# Exp 3: Parallel Training
model = SimpleNN(nn_size_template).to(device)

acc_df = log_accuracy('Parallel', 'Initial', acc_df, model, test_x, test_y, test_tasks)

train_network(model, train_x, train_y, opts)

acc_df = log_accuracy('Parallel', 'After Training', acc_df, model, test_x, test_y, test_tasks)

print(evaluate_per_task(model, test_x, test_y, test_tasks, num_tasks))
print(evaluate_all(model, test_x, test_y))

#%%
acc_df = pd.DataFrame(acc_df)

#%%
acc_df.to_csv(f'results.csv', index=False)

#%%