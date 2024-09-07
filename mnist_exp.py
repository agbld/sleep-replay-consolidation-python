#%%
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd

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

def create_class_task(x, y, task_size=2):
    """Splits the dataset into tasks by grouping certain classes."""
    tasks = np.zeros_like(y)
    unique_classes = np.unique(y)
    task_id = 0

    # Assign each pair of classes to a task
    for i in range(0, len(unique_classes), task_size):
        class_group = unique_classes[i:i+task_size]
        indices = np.isin(y, class_group)
        tasks[indices] = task_id
        task_id += 1

    return tasks, y

num_tasks = 5
task_size = (10 + num_tasks - 1) // num_tasks
print(f'Number of Tasks: {num_tasks}')
print(f'Task Size: {task_size}')
tasks, train_y = create_class_task(train_x, train_y, task_size)
test_tasks, test_y = create_class_task(test_x, test_y, task_size)

#%%
# Simple Neural Network

class SimpleNN(nn.Module):
    def __init__(self, layers):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x

def train_network(model, train_x, train_y, opts):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=opts['learning_rate'], momentum=opts['momentum'])
    criterion = nn.CrossEntropyLoss()
    num_epochs = opts['numepochs']
    batch_size = opts['batchsize']
    
    dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(torch.device(device)), target.to(torch.device(device))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return model

def evaluate_all(model, test_x, test_y):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    with torch.no_grad():  # Disable gradient calculations
        for data, target in loader:
            data, target = data.to(torch.device(device)), target.to(torch.device(device))
            output = model(data)
            _, predicted = torch.max(output.data, 1)  # Get the index of the max log-probability
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    # print(f'Accuracy All: {accuracy:.2f}%')
    return accuracy

def evaluate_per_task(model, test_x, test_y, test_tasks, num_tasks=num_tasks):
    model.eval()  # Set the model to evaluation mode
    accuracies = {}
    
    # Evaluate accuracy for each task
    with torch.no_grad():
        for task_id in range(num_tasks):
            # Get the indices of the examples for the current task
            task_indices = np.where(test_tasks == task_id)[0]
            task_test_x = test_x[task_indices]
            task_test_y = test_y[task_indices]
            
            # Create a DataLoader for this task's data
            dataset = TensorDataset(torch.tensor(task_test_x), torch.tensor(task_test_y, dtype=torch.long))
            loader = DataLoader(dataset, batch_size=100, shuffle=False)
            
            correct = 0
            total = 0
            for data, target in loader:
                data, target = data.to(torch.device(device)), target.to(torch.device(device))
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            accuracy = 100 * correct / total if total > 0 else 0
            accuracies[f'Task {task_id}'] = f'{accuracy:5.2f}'
            # print(f'Accuracy for Task {task_id}: {accuracy:.2f}%')
    
    return accuracies

def log_accuracy(approach: str, stage: str, acc_df: list, model: SimpleNN, test_x, test_y, test_tasks):
    acc_dict = {}
    acc_dict['Approach'] = approach
    acc_dict['Stage'] = stage
    acc_dict.update(evaluate_per_task(model, test_x, test_y, test_tasks))
    acc_dict['All'] = evaluate_all(model, test_x, test_y)
    acc_df.append(acc_dict)
    return acc_df

acc_df = []

opts = {
    'numepochs': 2,         # Number of epochs
    'batchsize': 100,       # Batch size for training
    'learning_rate': 0.1,   # Learning rate for SGD
    'momentum': 0.5         # Momentum for SGD
}

nn_size_template = [784, 4000, 4000, 10]

#%%
# Exp 1: Sleep Replay Consolidation (SRC)

def sleep_phase(nn: SimpleNN, num_iterations: int, sleep_opts: dict, sleep_input: torch.Tensor):
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
                    # Compute the input impulse based on spikes from the previous layer
                    impulse = nn.layers[l - 1](spikes[l - 1]) * sleep_opts['alpha'][l - 1]
                    impulse = impulse - torch.mean(impulse) * sleep_opts['W_inh'] # Apply inhibition

                    # Update the membrane potential with decay and the input impulse
                    membrane_potentials[l] = membrane_potentials[l] * sleep_opts['decay'] + impulse

                    # Add a direct current (DC) component for layer 4, if needed
                    # if l == len(nn_size) - 1:
                    #     membrane_potentials[l] += sleep_opts['DC']
                    
                    # Check for spiking based on membrane potential exceeding threshold
                    spikes[l] = torch.Tensor((membrane_potentials[l] >= sleep_opts['threshold'] * sleep_opts['beta'][l - 1]).float())



                    # Spike-Timing Dependent Plasticity (STDP) to adjust weights

                    # Get pre-synaptic spikes and post-synaptic spikes
                    pre = spikes[l - 1]  # (num_pre_neurons,)
                    post = spikes[l]  # (num_post_neurons,)

                    # Reshape pre and post for broadcasting
                    pre_broadcast = pre.unsqueeze(0)  # (1, num_pre_neurons)
                    post_broadcast = post.unsqueeze(1)  # (num_post_neurons, 1)

                    # Compute the weight delta using broadcasting
                    sigmoid_weights = torch.sigmoid(nn.layers[l - 1].weight)

                    # Apply increment where post is 1 and pre is 1
                    weight_inc = sleep_opts['inc'] * (post_broadcast == 1) * (pre_broadcast == 1) * sigmoid_weights

                    # Apply decrement where post is 1 and pre is 0
                    weight_dec = sleep_opts['dec'] * (post_broadcast == 1) * (pre_broadcast == 0) * sigmoid_weights

                    # Combine increments and decrements
                    weight_delta = weight_inc - weight_dec

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
        activations = nn.forward(torch.Tensor(x))  # Forward propagate through the network
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

def run_exp_3(sleep_iterations: int, acc_df: list):

    # src_model = SimpleNN([784, 1200, 1200, 10])
    src_model = SimpleNN(nn_size_template).to(device)

    # Define the hyperparameters for the sleep phase
    alpha_scale = 55.882454
    sleep_opts = {
        'beta': [14.548273, 44.560317, 38.046326],
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

    acc_df = log_accuracy(f'SRC-{sleep_iterations}', 'Initial', acc_df, src_model, test_x, test_y, test_tasks)

    for task_id in range(num_tasks):
        print(f'Task {task_id}')

        task_indices = np.where(tasks == task_id)[0]
        task_train_x = train_x[task_indices[:5000]]  # Use the first 5000 samples for each task
        task_train_y = train_y[task_indices[:5000]]
        
        src_model = train_network(src_model, task_train_x, task_train_y, opts)

        print('Before SRC: ', evaluate_per_task(src_model, test_x, test_y, test_tasks))
        
        acc_df = log_accuracy(f'SRC-{sleep_iterations}', 'Task ' + str(task_id) + ' Before SRC', acc_df, src_model, test_x, test_y, test_tasks)

        # Generate masked input for the sleep phase
        sleep_period = int(sleep_iterations + task_id * sleep_iterations/3)
        sleep_input = create_masked_input(task_train_x, sleep_period, 10)

        # Calculate the alpha
        # _, factor_log = normalize_nn_data(src_model, task_train_x)
        # sleep_opts['alpha'] = [alpha * alpha_scale for alpha in factor_log]
        # print('alpha: ', sleep_opts['alpha'])
        sleep_opts['alpha'] = [14.983829, 253.17746, 7.7707720]

        # Run the sleep phase
        src_model = sleep_phase(src_model, sleep_period, sleep_opts, torch.tensor(sleep_input))
        
        print('After SRC: ', evaluate_per_task(src_model, test_x, test_y, test_tasks))

        acc_df = log_accuracy(f'SRC-{sleep_iterations}', 'Task ' + str(task_id) + ' After SRC', acc_df, src_model, test_x, test_y, test_tasks)

    acc_df = log_accuracy(f'SRC-{sleep_iterations}', 'After Training', acc_df, src_model, test_x, test_y, test_tasks)

    print(evaluate_all(src_model, test_x, test_y))

    return acc_df

for sleep_iterations in [5, 20, 40, 60, 80, 100, 150, 200, 500, 1000]:
    acc_df = run_exp_3(sleep_iterations, acc_df)

# acc_df = run_exp_3(40, acc_df)

#%%
# Exp 2: Sequential Training
control_model = SimpleNN(nn_size_template).to(device)

acc_df = log_accuracy('Sequential', 'Initial', acc_df, control_model, test_x, test_y, test_tasks)

for task_id in range(num_tasks):
    task_indices = np.where(tasks == task_id)[0]
    task_train_x = train_x[task_indices[:5000]]  # Train on the first 5000 samples for each task
    task_train_y = train_y[task_indices[:5000]]
    
    control_model = train_network(control_model, task_train_x, task_train_y, opts)

    print(evaluate_per_task(control_model, test_x, test_y, test_tasks))

    acc_df = log_accuracy('Sequential', 'Task ' + str(task_id), acc_df, control_model, test_x, test_y, test_tasks)

acc_df = log_accuracy('Sequential', 'After Training', acc_df, control_model, test_x, test_y, test_tasks)

print(evaluate_all(control_model, test_x, test_y))

#%%
# Exp 3: Parallel Training
model = SimpleNN(nn_size_template).to(device)

acc_df = log_accuracy('Parallel', 'Initial', acc_df, model, test_x, test_y, test_tasks)

train_network(model, train_x, train_y, opts)

acc_df = log_accuracy('Parallel', 'After Training', acc_df, model, test_x, test_y, test_tasks)

print(evaluate_per_task(model, test_x, test_y, test_tasks))
print(evaluate_all(model, test_x, test_y))

#%%
acc_df = pd.DataFrame(acc_df)

#%%
acc_df.to_csv(f'results.csv', index=False)

#%%