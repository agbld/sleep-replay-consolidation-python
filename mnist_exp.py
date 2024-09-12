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
import copy

from utils.nn import SimpleNN, NeuronDeveloper
from utils.nn import train_network, evaluate_all, evaluate_per_task, log_accuracy, get_activations
from utils.task import create_class_task
from utils.sleep import create_masked_input

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

for i in range(10):
    task_indices = np.where(train_y == i)[0]
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

mean_pooling = True

#%%
# Exp 1: Sleep Replay Consolidation (SRC)

def sleep_phase(nn: SimpleNN, num_iterations: int, sleep_opts: dict, X: torch.Tensor):
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

        # --- Additional params from original SRC ---
        'bonus_iterations': 0,
        'mask_fraction': 0.25,
        'samples_per_iter': 10,
    }

    sleep_opts.update(sleep_opts_update)
    print(sleep_opts)

    acc_df = log_accuracy(f'SRC', 'Initial', acc_df, src_model, test_x, test_y, test_tasks, sleep_opts)

    for task_id in range(num_tasks):
        print(f'Task {task_id}')

        # [Visual] Initialize the NeuronDeveloper for layer visualization
        neuron_developer = NeuronDeveloper(title=f'Layer Activations for Task {task_id}: Before, After SRC, and Difference',
                                           output_path=f'./png/layer_activations_task_{task_id}_before_after_src.png')

        task_indices = np.where(train_tasks == task_id)[0]
        task_train_x = train_x[task_indices[:5000]]  # Use the first 5000 samples for each task
        task_train_y = train_y[task_indices[:5000]]
        
        model_before_training = copy.deepcopy(src_model)
        src_model = train_network(src_model, task_train_x, task_train_y, opts)

        # [Visual] Record activations before SRC for layer visualization
        neuron_developer.record(src_model, 
                                sequential_train_x, 
                                'Before SRC')

        print('Before SRC: ', evaluate_per_task(src_model, test_x, test_y, test_tasks, num_tasks))
        
        acc_df = log_accuracy(f'SRC', 'Task ' + str(task_id) + ' Before SRC', acc_df, src_model, test_x, test_y, test_tasks, sleep_opts)

        # Calculate the alpha
        # _, factor_log = normalize_nn_data(src_model, task_train_x)
        # sleep_opts['alpha'] = [alpha * sleep_opts['alpha_scale'] for alpha in factor_log]
        # print('alpha: ', sleep_opts['alpha'])

        # Run the sleep phase (with gradual increase in the number of iterations)
        sleep_period = int(sleep_opts['iterations'] + task_id * sleep_opts['bonus_iterations'])
        model_before_src = copy.deepcopy(src_model)
        src_model = sleep_phase(src_model, sleep_period, sleep_opts, task_train_x)
        
        print('After SRC: ', evaluate_per_task(src_model, test_x, test_y, test_tasks, num_tasks))

        acc_df = log_accuracy(f'SRC', 'Task ' + str(task_id) + ' After SRC', acc_df, src_model, test_x, test_y, test_tasks, sleep_opts)

        # [Visual] Record activations after SRC and calculate the difference
        neuron_developer.record(src_model, sequential_train_x, 'After SRC')

        # Create a synthetic model by using the model_before_training + (src_model - model_before_src)
        model_synthetic = SimpleNN(nn_size_template).to(device)
        with torch.no_grad():
            for i, layer in enumerate(model_synthetic.layers):
                with torch.no_grad():
                    layer.weight = torch.nn.Parameter(model_before_training.layers[i].weight + (src_model.layers[i].weight - model_before_src.layers[i].weight))

        print('Synthetic SRC: ', evaluate_per_task(model_synthetic, test_x, test_y, test_tasks, num_tasks))

        acc_df = log_accuracy(f'SRC', 'Task ' + str(task_id) + ' Synthetic SRC', acc_df, model_synthetic, test_x, test_y, test_tasks, sleep_opts)

        # [Visual] Record activations for the synthetic model
        neuron_developer.record(model_synthetic, sequential_train_x, 'Synthetic SRC')

        # [Visual] Record the difference between the activations before and after SRC
        neuron_developer.record_diff('After SRC', 'Before SRC', 'Difference')

        # [Visual] Reduce the dimensionality of the activations using PCA
        neuron_developer.reduce(pca_components=10)

        # [Visual] Show and save the plot
        neuron_developer.show(mean_pooling)
        neuron_developer.save()

    acc_df = log_accuracy(f'SRC', 'After Training', acc_df, src_model, test_x, test_y, test_tasks, sleep_opts)

    print(evaluate_all(src_model, test_x, test_y))

    return acc_df

for iteration in [400]:
    for mask_fraction in [0.25]:
        acc_df = run_sleep_exp(
            acc_df, 
            {
                'iterations': iteration, 
                'inc': 0.001,
                'dec': 0.0001,

                # --- Additional params from original SRC ---
                'bonus_iterations': int(iteration / 3),
                'mask_fraction': mask_fraction, # original: 0.25 (aprox.)
                'samples_per_iter': 10, # original: (entire X from current task)
            },)

#%%
# Exp 2: Model Merging

task_models = []

# [Visual] Initialize the NeuronDeveloper for layer visualization
neuron_developer = NeuronDeveloper(title=f'Layer Activations: Merging',
                                    output_path=f'./png/layer_activations_merging.png')

for task_id in range(num_tasks):

    task_model = SimpleNN(nn_size_template).to(device)
    task_indices = np.where(train_tasks == task_id)[0]
    task_train_x = train_x[task_indices[:5000]]  # Use the first 5000 samples for each task
    task_train_y = train_y[task_indices[:5000]]
    acc_df = log_accuracy('Merging', 'Task ' + str(task_id) + ' Initial', acc_df, task_model, test_x, test_y, test_tasks)
    task_model = train_network(task_model, task_train_x, task_train_y, opts)
    print(evaluate_per_task(task_model, test_x, test_y, test_tasks, num_tasks))
    acc_df = log_accuracy('Merging', 'Task ' + str(task_id), acc_df, task_model, test_x, test_y, test_tasks)

    # [Visual] Record activations layer visualization
    neuron_developer.record(task_model, sequential_train_x, 'Task ' + str(task_id) + ' Model')

    task_models.append(task_model)

merged_model = SimpleNN(nn_size_template).to(device)
for layer in merged_model.layers:
    if hasattr(layer, 'weight'):
        torch.nn.init.constant_(layer.weight, 0)
    if hasattr(layer, 'bias') and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, 0)

for task_model in task_models:
    for i, layer in enumerate(task_model.layers):
        with torch.no_grad():
            merged_model.layers[i].weight += layer.weight / num_tasks
            if layer.bias is not None:
                merged_model.layers[i].bias += layer.bias / num_tasks

acc_df = log_accuracy('Merging', 'Merged', acc_df, merged_model, test_x, test_y, test_tasks)
print(evaluate_per_task(merged_model, test_x, test_y, test_tasks, num_tasks))
print(evaluate_all(merged_model, test_x, test_y))

# [Visual] Record activations layer visualization
neuron_developer.record(merged_model, sequential_train_x, 'Merged')

# [Visual] Reduce the dimensionality of the activations using PCA
neuron_developer.reduce(pca_components=10)

# [Visual] Show and save the plot
neuron_developer.show(mean_pooling)
neuron_developer.save()

#%%
# Exp 3: Sequential Training
control_model = SimpleNN(nn_size_template).to(device)

acc_df = log_accuracy('Sequential', 'Initial', acc_df, control_model, test_x, test_y, test_tasks)

for task_id in range(num_tasks):

    # [Visual] Initialize the NeuronDeveloper for layer visualization
    neuron_developer = NeuronDeveloper(title=f'Layer Activations for Task {task_id}: Sequential',
                                       output_path=f'./png/layer_activations_task_{task_id}_sequential.png')
    
    task_indices = np.where(train_tasks == task_id)[0]
    task_train_x = train_x[task_indices[:5000]]  # Train on the first 5000 samples for each task
    task_train_y = train_y[task_indices[:5000]]
    
    control_model = train_network(control_model, task_train_x, task_train_y, opts)

    print(evaluate_per_task(control_model, test_x, test_y, test_tasks, num_tasks))

    acc_df = log_accuracy('Sequential', 'Task ' + str(task_id), acc_df, control_model, test_x, test_y, test_tasks)

    # [Visual] Record activations layer visualization
    neuron_developer.record(control_model, sequential_train_x, 'Sequential')

    # [Visual] Reduce the dimensionality of the activations using PCA
    neuron_developer.reduce(pca_components=10)

    # [Visual] Show and save the plot
    neuron_developer.show(mean_pooling)
    neuron_developer.save()

acc_df = log_accuracy('Sequential', 'After Training', acc_df, control_model, test_x, test_y, test_tasks)

print(evaluate_all(control_model, test_x, test_y))

#%%
# Exp 4: Parallel Training
parallel_model = SimpleNN(nn_size_template).to(device)

# [Visual] Initialize the NeuronDeveloper for layer visualization
neuron_developer = NeuronDeveloper(title=f'Layer Activations: Parallel',
                                   output_path=f'./png/layer_activations_parallel.png')

acc_df = log_accuracy('Parallel', 'Initial', acc_df, parallel_model, test_x, test_y, test_tasks)

train_network(parallel_model, train_x, train_y, opts)

acc_df = log_accuracy('Parallel', 'After Training', acc_df, parallel_model, test_x, test_y, test_tasks)

print(evaluate_per_task(parallel_model, test_x, test_y, test_tasks, num_tasks))
print(evaluate_all(parallel_model, test_x, test_y))

# [Visual] Record activations layer visualization
neuron_developer.record(parallel_model, sequential_train_x, 'Parallel')

# [Visual] Reduce the dimensionality of the activations using PCA
neuron_developer.reduce(pca_components=10)

# [Visual] Show and save the plot
neuron_developer.show(mean_pooling)
neuron_developer.save()

#%%
acc_df = pd.DataFrame(acc_df)

#%%
acc_df.to_csv(f'results.csv', index=False)

#%%