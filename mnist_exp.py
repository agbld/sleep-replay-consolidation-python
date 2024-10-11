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
from utils.task import create_class_task, create_permutation_task

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# MNIST Dataset Loading

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Convert to numpy format for task generation (optional)
train_X = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255
train_Y = train_dataset.targets.numpy()
test_X = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255
test_Y = test_dataset.targets.numpy()

#%%
# Task Generation (Class-based)

# num_tasks = 5
# print(f'Number of Tasks: {num_tasks}')

# # Class-based task generation
# task_size = (10 + num_tasks - 1) // num_tasks
# print(f'Task Size: {task_size}')
# train_X_list, train_Y_list = create_class_task(train_X, train_Y, task_size, num_tasks)
# test_X_list, test_Y_list = create_class_task(test_X, test_Y, task_size, num_tasks)

# # Keep only the first 5000 samples for each task
# for i in range(num_tasks):
#     train_X_list[i] = train_X_list[i][:5000]
#     train_Y_list[i] = train_Y_list[i][:5000]

# Create sequential x for visualization
train_X_sequential = []
for i in range(10):
    class_id = np.where(train_Y == i)[0]
    X_class = train_X[class_id[:1000]]
    train_X_sequential.append(X_class)
train_X_sequential = np.concatenate(train_X_sequential)

#%%
# Task Generation (Permutation-based)

num_tasks = 5
print(f'Number of Tasks: {num_tasks}')

# Permutation-based task generation
train_X_list, train_Y_list, used_perms = create_permutation_task(train_X, train_Y, num_tasks)
test_X_list, test_Y_list, _ = create_permutation_task(test_X, test_Y, num_tasks, perms=used_perms) # Use the same permutations for test tasks

# Keep only the first 5000 samples for each task
for i in range(num_tasks):
    train_X_list[i] = train_X_list[i][:50000]
    train_Y_list[i] = train_Y_list[i][:50000]

# Concatenate all tasks for parallel training
train_X = np.concatenate(train_X_list)
train_Y = np.concatenate(train_Y_list)
test_X = np.concatenate(test_X_list)
test_Y = np.concatenate(test_Y_list)

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

    # sleep_opts['stable_ranks'] = compute_stable_rank(src_model)
    # print("Stable Ranks:", sleep_opts['stable_ranks'])
    acc_df = log_accuracy(f'SRC', 'Initial', acc_df, src_model, test_X_list, test_Y_list, sleep_opts)

    for task_id in range(num_tasks):
        print(f'Task {task_id}')

        # # [Visual] Initialize the NeuronDeveloper for layer visualization
        # neuron_developer = NeuronDeveloper(title=f'Layer Activations for Task {task_id}: Before, After SRC, and Difference',
        #                                    output_path=f'./png/layer_activations_task_{task_id}_before_after_src.png')

        train_X_task = train_X_list[task_id]
        train_Y_task = train_Y_list[task_id]
        
        # model_before_training = copy.deepcopy(src_model) # Take a snapshot of the model before training (for synthetic model creation)
        src_model = train_network(src_model, train_X_task, train_Y_task, opts)

        # # [Visual] Record activations before SRC for layer visualization
        # neuron_developer.record(src_model, 
        #                         train_X_sequential, 
        #                         'Before SRC')

        print('Before SRC: ', evaluate_per_task(src_model, test_X_list, test_Y_list))
        # sleep_opts['stable_ranks'] = compute_stable_rank(src_model)
        # print("Stable Ranks:", sleep_opts['stable_ranks'])
        acc_df = log_accuracy(f'SRC', 'Task ' + str(task_id) + ' Before SRC', acc_df, src_model, test_X_list, test_Y_list, sleep_opts)

        # Calculate the alpha
        # _, factor_log = normalize_nn_data(src_model, task_train_x)
        # sleep_opts['alpha'] = [alpha * sleep_opts['alpha_scale'] for alpha in factor_log]
        # print('alpha: ', sleep_opts['alpha'])

        # Run the sleep phase (with gradual increase in the number of iterations)
        sleep_period = int(sleep_opts['iterations'] + task_id * sleep_opts['bonus_iterations'])
        # model_before_src = copy.deepcopy(src_model) # Take a snapshot of the model before SRC (for synthetic model creation)
        src_model = sleep_phase(src_model, sleep_period, sleep_opts, train_X_task)
        
        print('After SRC: ', evaluate_per_task(src_model, test_X_list, test_Y_list))
        # sleep_opts['stable_ranks'] = compute_stable_rank(src_model)
        # print("Stable Ranks:", sleep_opts['stable_ranks'])
        acc_df = log_accuracy(f'SRC', 'Task ' + str(task_id) + ' After SRC', acc_df, src_model, test_X_list, test_Y_list, sleep_opts)

        # # [Visual] Record activations after SRC and calculate the difference
        # neuron_developer.record(src_model, train_X_sequential, 'After SRC')

        # # Create a synthetic model by using the model_before_training + (src_model - model_before_src)
        # model_synthetic = SimpleNN(nn_size_template).to(device)
        # with torch.no_grad():
        #     for i, layer in enumerate(model_synthetic.layers):
        #         with torch.no_grad():
        #             layer.weight = torch.nn.Parameter(model_before_training.layers[i].weight + (src_model.layers[i].weight - model_before_src.layers[i].weight))

        # print('Synthetic SRC: ', evaluate_per_task(model_synthetic, test_X_list, test_Y_list))
        # sleep_opts['stable_ranks'] = [compute_stable_rank(src_model)]
        # print("Stable Ranks:", sleep_opts['stable_ranks'])
        # acc_df = log_accuracy(f'SRC', 'Task ' + str(task_id) + ' Synthetic SRC', acc_df, model_synthetic, test_X_list, test_Y_list, sleep_opts)

        # # [Visual] Record activations for the synthetic model
        # neuron_developer.record(model_synthetic, train_X_cat, 'Synthetic SRC')

        # # [Visual] Record the difference between the activations before and after SRC
        # neuron_developer.record_diff('After SRC', 'Before SRC', 'Difference')

        # # [Visual] Reduce the dimensionality of the activations using PCA
        # neuron_developer.reduce(pca_components=10)

        # # [Visual] Show and save the plot
        # neuron_developer.show(mean_pooling)
        # neuron_developer.save()

    # sleep_opts['stable_ranks'] = compute_stable_rank(src_model)
    # print("Stable Ranks:", sleep_opts['stable_ranks'])
    acc_df = log_accuracy(f'SRC', 'After Training', acc_df, src_model, test_X_list, test_Y_list, sleep_opts)

    print(evaluate_all(src_model, test_X, test_Y))

    return acc_df

for iteration in [10, 25, 50, 100, 250, 500, 1000]:
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

# # [Visual] Initialize the NeuronDeveloper for layer visualization
# neuron_developer = NeuronDeveloper(title=f'Layer Activations: Merging',
#                                     output_path=f'./png/layer_activations_merging.png')

for task_id in range(num_tasks):

    task_model = SimpleNN(nn_size_template).to(device)
    train_X_task = train_X_list[task_id]
    train_Y_task = train_Y_list[task_id]
    acc_df = log_accuracy('Merging', 'Task ' + str(task_id) + ' Initial', acc_df, task_model, test_X_list, test_Y_list)
    task_model = train_network(task_model, train_X_task, train_Y_task, opts)
    print(evaluate_per_task(task_model, test_X_list, test_Y_list))
    acc_df = log_accuracy('Merging', 'Task ' + str(task_id), acc_df, task_model, test_X_list, test_Y_list)

    # # [Visual] Record activations layer visualization
    # neuron_developer.record(task_model, train_X_sequential, 'Task ' + str(task_id) + ' Model')

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

acc_df = log_accuracy('Merging', 'Merged', acc_df, merged_model, test_X_list, test_Y_list)
print(evaluate_per_task(merged_model, test_X_list, test_Y_list))
print(evaluate_all(merged_model, test_X, test_Y))

# # [Visual] Record activations layer visualization
# neuron_developer.record(merged_model, train_X_sequential, 'Merged')

# # [Visual] Reduce the dimensionality of the activations using PCA
# neuron_developer.reduce(pca_components=10)

# # [Visual] Show and save the plot
# neuron_developer.show(mean_pooling)
# neuron_developer.save()

#%%
# Exp 3: Sequential Training
control_model = SimpleNN(nn_size_template).to(device)

acc_df = log_accuracy('Sequential', 'Initial', acc_df, control_model, test_X_list, test_Y_list)

for task_id in range(num_tasks):

    # # [Visual] Initialize the NeuronDeveloper for layer visualization
    # neuron_developer = NeuronDeveloper(title=f'Layer Activations for Task {task_id}: Sequential',
    #                                    output_path=f'./png/layer_activations_task_{task_id}_sequential.png')
    
    train_X_task = train_X_list[task_id]
    train_Y_task = train_Y_list[task_id]
    
    control_model = train_network(control_model, train_X_task, train_Y_task, opts)

    print(evaluate_per_task(control_model, test_X_list, test_Y_list))

    acc_df = log_accuracy('Sequential', 'Task ' + str(task_id), acc_df, control_model, test_X_list, test_Y_list)

    # # [Visual] Record activations layer visualization
    # neuron_developer.record(control_model, train_X_sequential, 'Sequential')

    # # [Visual] Reduce the dimensionality of the activations using PCA
    # neuron_developer.reduce(pca_components=10)

    # # [Visual] Show and save the plot
    # neuron_developer.show(mean_pooling)
    # neuron_developer.save()

acc_df = log_accuracy('Sequential', 'After Training', acc_df, control_model, test_X_list, test_Y_list)

print(evaluate_all(control_model, test_X, test_Y))

#%%
# Exp 4: Parallel Training
parallel_model = SimpleNN(nn_size_template).to(device)

# # [Visual] Initialize the NeuronDeveloper for layer visualization
# neuron_developer = NeuronDeveloper(title=f'Layer Activations: Parallel',
#                                    output_path=f'./png/layer_activations_parallel.png')

acc_df = log_accuracy('Parallel', 'Initial', acc_df, parallel_model, test_X_list, test_Y_list)

train_network(parallel_model, train_X, train_Y, opts)

acc_df = log_accuracy('Parallel', 'After Training', acc_df, parallel_model, test_X_list, test_Y_list)

print(evaluate_per_task(parallel_model, test_X_list, test_Y_list))
print(evaluate_all(parallel_model, test_X, test_Y))

# # [Visual] Record activations layer visualization
# neuron_developer.record(parallel_model, train_X_sequential, 'Parallel')

# # [Visual] Reduce the dimensionality of the activations using PCA
# neuron_developer.reduce(pca_components=10)

# # [Visual] Show and save the plot
# neuron_developer.show(mean_pooling)
# neuron_developer.save()

#%%
acc_df = pd.DataFrame(acc_df)

#%%
acc_df.to_csv(f'results.csv', index=False)

#%%