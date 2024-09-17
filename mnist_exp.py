#%%
# Import necessary libraries
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import copy

from utils.nn import SimpleNN, NeuronDeveloper
from utils.nn import train_network, evaluate_all, evaluate_per_task, log_accuracy, compute_stable_rank
from utils.sleep import sleep_phase
from utils.task import create_class_task

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

nn_size_template = [784, 1200, 1200, 10]

mean_pooling = True

#%%
# Exp 1: Sleep Replay Consolidation (SRC)

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

    sleep_opts['stable_ranks'] = compute_stable_rank(src_model)
    acc_df = log_accuracy(f'SRC', 'Initial', acc_df, src_model, test_x, test_y, test_tasks, sleep_opts)

    for task_id in range(num_tasks):
        print(f'Task {task_id}')

        # [Visual] Initialize the NeuronDeveloper for layer visualization
        neuron_developer = NeuronDeveloper(title=f'Layer Activations for Task {task_id}: Before, After SRC, and Difference',
                                           output_path=f'./png/layer_activations_task_{task_id}_before_after_src.png')

        task_indices = np.where(train_tasks == task_id)[0]
        task_train_x = train_x[task_indices[:5000]]  # Use the first 5000 samples for each task
        task_train_y = train_y[task_indices[:5000]]
        
        model_before_training = copy.deepcopy(src_model) # Take a snapshot of the model before training (for synthetic model creation)
        src_model = train_network(src_model, task_train_x, task_train_y, opts)

        # [Visual] Record activations before SRC for layer visualization
        neuron_developer.record(src_model, 
                                sequential_train_x, 
                                'Before SRC')

        print('Before SRC: ', evaluate_per_task(src_model, test_x, test_y, test_tasks, num_tasks))
        sleep_opts['stable_ranks'] = compute_stable_rank(src_model)
        acc_df = log_accuracy(f'SRC', 'Task ' + str(task_id) + ' Before SRC', acc_df, src_model, test_x, test_y, test_tasks, sleep_opts)

        # Calculate the alpha
        # _, factor_log = normalize_nn_data(src_model, task_train_x)
        # sleep_opts['alpha'] = [alpha * sleep_opts['alpha_scale'] for alpha in factor_log]
        # print('alpha: ', sleep_opts['alpha'])

        # Run the sleep phase (with gradual increase in the number of iterations)
        sleep_period = int(sleep_opts['iterations'] + task_id * sleep_opts['bonus_iterations'])
        model_before_src = copy.deepcopy(src_model) # Take a snapshot of the model before SRC (for synthetic model creation)
        src_model = sleep_phase(src_model, sleep_period, sleep_opts, task_train_x)
        
        print('After SRC: ', evaluate_per_task(src_model, test_x, test_y, test_tasks, num_tasks))
        sleep_opts['stable_ranks'] = compute_stable_rank(src_model)
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
        sleep_opts['stable_ranks'] = [compute_stable_rank(src_model)]
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

    sleep_opts['stable_ranks'] = compute_stable_rank(src_model)
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