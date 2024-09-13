import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleNN(nn.Module):
    def __init__(self, layers):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1], bias=False) for i in range(len(layers) - 1)])
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

def evaluate_per_task(model, test_x, test_y, test_tasks, num_tasks):
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

def log_accuracy(approach: str, stage: str, acc_df: list, model: SimpleNN, test_x, test_y, test_tasks, args: dict = {}):
    acc_dict = {}
    acc_dict['Approach'] = approach
    acc_dict['Stage'] = stage
    acc_dict.update(args)
    acc_dict.update(evaluate_per_task(model, test_x, test_y, test_tasks, len(set(test_tasks))))
    acc_dict['All'] = evaluate_all(model, test_x, test_y)
    acc_df.append(acc_dict)
    return acc_df

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

class NeuronDeveloper():
    def __init__(self,
                 title: str, 
                 output_path: str, 
                 get_activations=get_activations,
                 device=None):
        self.title = title
        self.output_path = output_path
        self.get_activations = get_activations
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.activations = {}
        self.reduced_activations = {}
        self.fig = None
        
    def record(self, model: SimpleNN, x, subtitle: str) -> list:
        with torch.no_grad():
            model.eval()
            activation = self.get_activations(model.to(self.device), torch.Tensor(x).to(self.device))
            activation = [v.cpu() for v in activation.values()]
            self.activations[subtitle] = activation

        return activation
    
    def record_diff(self, post_subtitle: str, pre_subtitle: str, subtitle: str = None) -> list:
        if not subtitle:
            subtitle = post_subtitle + ' - ' + pre_subtitle
        self.activations[subtitle] = [post - pre for post, pre in zip(self.activations[post_subtitle], self.activations[pre_subtitle])]
        return self.activations[subtitle]

    def reduce(self, reduction_method='tsne', components=[None, None, None]):
        # Get the number of layers from the first entry in activations
        num_layers = len(next(iter(self.activations.values())))

        # Precompute models for each layer except the last one
        reduction_models = [None] * num_layers

        # Pre-fetch all the activations for improved loop performance
        activations_values = {subtitle: [a.to(self.device) for a in activations] for subtitle, activations in self.activations.items()}

        for i in range(num_layers):
            # Concatenate activations across all subtitles for layer i
            layer_activations = [activation[i].cpu().numpy() for activation in activations_values.values()]

            # Combine activations into a single array for dimensionality reduction
            concat_activations = np.concatenate(layer_activations, axis=0)

            if components[i]:  # Only reduce if components are specified
                if reduction_models[i] is None:
                    if reduction_method == 'pca':
                        reduction_models[i] = PCA(n_components=components[i])
                        reduction_models[i].fit(concat_activations)
                    elif reduction_method == 'tsne':
                        # Note: t-SNE does not require a fit step, so we do fit_transform directly
                        reduction_models[i] = TSNE(n_components=components[i], random_state=42)
                        concat_activations = reduction_models[i].fit_transform(concat_activations)  # Apply t-SNE to the concatenated activations
                
            # Transform the activations for each subtitle
            for subtitle, activation in self.activations.items():
                if self.reduced_activations.get(subtitle) is None:
                    self.reduced_activations[subtitle] = []

                layer_activation = activation[i].cpu().numpy()

                if components[i]:
                    if reduction_method == 'pca':
                        reduced_activation = reduction_models[i].transform(layer_activation)
                    elif reduction_method == 'tsne':
                        # For t-SNE, the transformation happens on the full dataset, so you cannot apply it like PCA
                        # Instead, the `concat_activations` are already reduced, so we split them back
                        reduced_activation = concat_activations[:len(layer_activation)]
                        concat_activations = concat_activations[len(layer_activation):]  # Update remaining activations
                else:
                    reduced_activation = layer_activation  # No dimensionality reduction

                # Append the reduced activation for the current layer
                self.reduced_activations[subtitle].append(reduced_activation)

        return self.reduced_activations

    
    def render(self, plot_types: list = ['heatmap', 'heatmap', 'heatmap'], pooling_types=['mean', 'mean', 'mean']):
        num_layers = len(self.activations[list(self.activations.keys())[0]])

        fig, axs = plt.subplots(len(self.reduced_activations.keys()), 
                                num_layers, 
                                figsize=(num_layers * 6, 
                                         len(self.reduced_activations.keys()) * 4 + 1))
        
        fig.suptitle(self.title, fontsize=24)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        for i in range(num_layers):
            if len(self.reduced_activations.keys()) == 1:
                # Do pooling if mean_pooling is True
                data = self.reduced_activations[list(self.reduced_activations.keys())[0]][i]
                pca_components = data.shape[1]
                if pooling_types[i] == 'mean':
                    pooled_data = data.reshape(-1, 1000, pca_components).mean(axis=1)
                elif pooling_types[i] == 'none':
                    pooled_data = data
                
                # Plot the figure according to the plot_types[i]
                if plot_types[i] == 'heatmap':
                    im = axs[i].imshow(pooled_data, aspect='auto')
                elif plot_types[i] == 'scatter':
                    if pca_components and pca_components != 2:
                        print("Warning: pca_components is not 2. Scatter plot may not work as expected.")
                    colors = np.repeat(np.arange(pooled_data.shape[0] // 1000), 1000)
                    im = axs[i].scatter(pooled_data[:, 0], pooled_data[:, 1], c=colors, cmap='viridis')

                axs[i].set_title(list(self.reduced_activations.keys())[0] + f' - Layer {i}', fontsize=18)
                fig.colorbar(im, ax=axs[i])
            else:
                for j, sub_title in enumerate(self.reduced_activations.keys()):
                    # Do pooling if mean_pooling is True
                    data = self.reduced_activations[sub_title][i]
                    pca_components = data.shape[1]
                    if pooling_types[i] == 'mean':
                        pooled_data = data.reshape(-1, 1000, pca_components).mean(axis=1)
                    elif pooling_types[i] == 'none':
                        pooled_data = data

                    # Plot the figure according to the plot_types[i]
                    if plot_types[i] == 'heatmap':
                        im = axs[j, i].imshow(pooled_data, aspect='auto')
                    elif plot_types[i] == 'scatter':
                        if pca_components and pca_components != 2:
                            print("Warning: pca_components is not 2. Scatter plot may not work as expected.")
                        colors = np.repeat(np.arange(pooled_data.shape[0] // 1000), 1000)
                        im = axs[j, i].scatter(pooled_data[:, 0], pooled_data[:, 1], c=colors, cmap='viridis')

                    axs[j, i].set_title(sub_title + f' - Layer {i}', fontsize=18)
                    fig.colorbar(im, ax=axs[j, i])

        # Adjust spacing between rows
        plt.subplots_adjust(hspace=0.2)
        
        self.fig = fig

    def save(self):
        self.fig.savefig(self.output_path, bbox_inches='tight', facecolor='w')