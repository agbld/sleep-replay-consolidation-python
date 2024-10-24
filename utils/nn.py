import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

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

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LowRankLinear, self).__init__()
        self.rank = rank
        # Define the two lower-rank matrices
        self.A = nn.Linear(rank, out_features, bias=False)  # A: (out_features x rank)
        self.B = nn.Linear(in_features, rank, bias=False)   # B: (rank x in_features)

    def forward(self, x):
        # Perform the low-rank linear transformation
        return self.A(self.B(x))
    
    def __add__(self, other):
        if not isinstance(other, LowRankLinear):
            raise ValueError("Both operands must be instances of LowRankLinear")
        
        # Concatenate the A matrices and B matrices
        A_prime = torch.cat((self.A.weight.data, other.A.weight.data), dim=1)
        B_prime = torch.cat((self.B.weight.data, other.B.weight.data), dim=0)
        
        # Create a new LowRankLinear layer with the concatenated matrices
        new_layer = LowRankLinear(self.B.in_features, self.A.out_features, self.rank + other.rank)
        new_layer.A.weight.data = A_prime
        new_layer.B.weight.data = B_prime
        
        return new_layer

class LowRankNN(nn.Module):
    def __init__(self, layers, max_rank):
        super(LowRankNN, self).__init__()
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

        # Initialize low-rank linear layers
        for i in range(len(layers) - 1):
            self.layers.append(LowRankLinear(layers[i], layers[i + 1], max_rank))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)  # Apply low-rank linear layer
            x = self.relu(x)
            x = self.dropout(x)
        
        # For the last layer, no ReLU or Dropout
        x = self.layers[-1](x)
        return x

    def get_simple_nn(self):
        # Convert the low-rank model to a simple model
        simple_layers = [layer.A.weight.data.shape[1] for layer in self.layers]
        simple_layers.append(self.layers[-1].A.weight.data.shape[0])
        simple_nn = SimpleNN(simple_layers)
        
        # Copy the weights from the low-rank model to the simple model
        for i, layer in enumerate(simple_nn.layers):
            layer.weight.data = self.layers[i].A.weight.data @ self.layers[i].B.weight.data
        
        return simple_nn
    
    def __add__(self, other):
        if not isinstance(other, LowRankNN):
            raise ValueError("Both operands must be instances of LowRankNN")
        if len(self.layers) != len(other.layers):
            raise ValueError("Both LowRankNN models must have the same number of layers")

        new_layers = []
        for self_layer, other_layer in zip(self.layers, other.layers):
            if self_layer.A.out_features != other_layer.A.out_features or self_layer.B.in_features != other_layer.B.in_features:
                raise ValueError("Both LowRankNN models must have the same number of neurons in each layer")
            new_layers.append(self_layer + other_layer)

        new_model = LowRankNN([layer.B.in_features for layer in self.layers] + [self.layers[-1].A.out_features], max_rank=self.layers[0].rank + other.layers[0].rank)
        new_model.layers = nn.ModuleList(new_layers)
        return new_model

def train_network(model, train_x, train_y, opts, verbose=False):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=opts['learning_rate'], momentum=opts['momentum'])
    criterion = nn.CrossEntropyLoss()
    num_epochs = opts['numepochs']
    batch_size = opts['batchsize']
    
    dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    with tqdm(total=num_epochs * len(loader), disable=not verbose) as pbar:
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(torch.device(device)), target.to(torch.device(device))
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                pbar.update(1)

    return model

def evaluate_all(model, X, Y):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y, dtype=torch.long))
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

def evaluate_per_task(model, X_list, Y_list, current_task=None):
    model.eval()  # Set the model to evaluation mode
    accuracies = {}
    num_tasks = len(X_list)
    if num_tasks != len(Y_list):
        raise ValueError('Number of task inputs and outputs must match')
    
    if current_task is not None:
        current_task_cls = list(set(Y_list[current_task]))
    
    # Evaluate accuracy for each task
    with torch.no_grad():
        for task_id in range(num_tasks):
            # Get the indices of the examples for the current task
            task_test_x = X_list[task_id]
            task_test_y = Y_list[task_id]
            
            # Create a DataLoader for this task's data
            dataset = TensorDataset(torch.tensor(task_test_x), torch.tensor(task_test_y, dtype=torch.long))
            loader = DataLoader(dataset, batch_size=100, shuffle=False)
            
            correct = 0
            total = 0
            for data, target in loader:
                data, target = data.to(torch.device(device)), target.to(torch.device(device))
                output = model(data)
                
                # Do "cheat" if current_task is not None
                if current_task is not None:
                    if current_task != task_id:
                        # Set the `output` to 0 for each neuron with index not in the current_task_classes
                        for i in range(len(output)):
                            output[i][[j for j in range(10) if j in current_task_cls]] = 0

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            accuracy = 100 * correct / total if total > 0 else 0
            accuracies[f'Task {task_id}'] = f'{accuracy:5.2f}'

    # Calculate the average accuracy across all tasks (weighted by the number of examples)
    total_examples = sum([len(Y) for Y in Y_list])
    weighted_accuracies = [len(Y_list[i]) * float(acc) for i, acc in enumerate(accuracies.values())]
    avg_accuracy = sum(weighted_accuracies) / total_examples
    accuracies['Task All'] = f'{avg_accuracy:5.2f}'
    
    return accuracies

def log_accuracy(approach: str, stage: str, acc_df: list, evaluate_results, args: dict = {}):
    acc_dict = {}
    acc_dict['Approach'] = approach
    acc_dict['Stage'] = stage
    acc_dict.update(args)
    acc_dict.update(evaluate_results)
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
                 device=None,
                 disable=False):
        self.title = title
        self.output_path = output_path
        self.get_activations = get_activations
        self.disable = disable
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.activations = {}
        self.reduced_activations = {}
        self.fig = None
        
    def record(self, model: SimpleNN, x, subtitle: str) -> list:
        if self.disable:
            return []
        with torch.no_grad():
            model.eval()
            activation = self.get_activations(model.to(self.device), torch.Tensor(x).to(self.device))
            activation = [v.cpu() for v in activation.values()]
            self.activations[subtitle] = activation

        return activation
    
    def record_diff(self, post_subtitle: str, pre_subtitle: str, subtitle: str = None) -> list:
        if self.disable:
            return []
        if not subtitle:
            subtitle = post_subtitle + ' - ' + pre_subtitle
        self.activations[subtitle] = [post - pre for post, pre in zip(self.activations[post_subtitle], self.activations[pre_subtitle])]
        return self.activations[subtitle]
    
    def show_activation_difference(self, post_subtitle: str, pre_subtitle: str, task_id: int, output_path: str = None):
        if self.disable:
            return
        act_post = self.activations[post_subtitle]
        act_pre = self.activations[pre_subtitle]
        
        # Calculate the cosine similarity between the activations before and after SRC
        act_cosine = []
        for i in range(len(act_post)):
            act_cosine_layer = []
            for j in range(10):
                cosine_class = torch.nn.functional.cosine_similarity(act_post[i][j * 1000: (j + 1) * 1000], 
                                                                     act_pre[i][j * 1000: (j + 1) * 1000], 
                                                                     dim=1)
                cosine_class = cosine_class.mean().item()
                act_cosine_layer.append(cosine_class)
            act_cosine.append(act_cosine_layer)
        
        # layer wise normalization for act_cosine
        act_cosine = np.array(act_cosine)
        # act_cosine = (act_cosine - act_cosine.min(axis=1)[:, None]) / (act_cosine.max(axis=1) - act_cosine.min(axis=1))[:, None]

        # class wise mean for act_cosine
        act_cosine_mean = act_cosine.mean(axis=0)

        # Create a figure with 2 subplots: one for the difference heatmap and one for the mean cosine similarity heatmap
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 5), 
                           gridspec_kw={'height_ratios': [3, 0.5]})  # Adjust the height ratio

        # Plot the difference heatmap (1 - cosine similarity)
        act_diff = 1 - np.array(act_cosine)
        cax1 = ax1.imshow(act_diff, cmap='Reds', interpolation='nearest', aspect='auto')
        fig.colorbar(cax1, ax=ax1)
        ax1.set_title(f'Layer Activations Difference For Each Class at Task {task_id} (Normalized)', fontsize=18)
        ax1.set(ylabel='Layer', xlabel='Class', 
            yticks=[0, 1, 2], xticks=range(10), 
            xticklabels=[str(i) for i in range(10)])

        # Annotate the heatmap with the values
        for i in range(act_diff.shape[0]):
            for j in range(act_diff.shape[1]):
                ax1.text(j, i, f'{act_diff[i, j]:.2f}', ha='center', va='center', color='black')

        # Plot the mean difference heatmap (1 - cosine similarity)
        act_diff_mean = 1 - np.array(act_cosine_mean)
        cax2 = ax2.imshow(act_diff_mean.reshape(1, -1), cmap='Reds', interpolation='nearest', aspect='auto')
        fig.colorbar(cax2, ax=ax2)
        ax2.set_title(f'Mean Layer Activations Difference For Each Class at Task {task_id} (Normalized)', fontsize=18)
        ax2.set(ylabel='Layer', xlabel='Class', 
            yticks=[0], xticks=range(10), 
            xticklabels=[str(i) for i in range(10)])

        # Annotate the mean heatmap with the values
        for j in range(act_diff_mean.shape[0]):
            ax2.text(j, 0, f'{act_diff_mean[j]:.2f}', ha='center', va='center', color='black')
        
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

        if output_path:
            fig.savefig(output_path, bbox_inches='tight', facecolor='w')

    def reduce(self, pca_components=None):
        if self.disable:
            return {}
        # Get the number of layers from the first entry in activations
        num_layers = len(next(iter(self.activations.values())))

        # Precompute PCA models for each layer except the last one
        pca_models = [None] * num_layers

        # Pre-fetch all the activations for improved loop performance
        activations_values = {subtitle: [a.to(self.device) for a in activations] for subtitle, activations in self.activations.items()}

        for i in range(num_layers):
            # Concatenate activations across all subtitles for layer i
            layer_activations = [activation[i].cpu().numpy() for activation in activations_values.values()]

            # Combine activations into a single array for PCA
            concat_activations = np.concatenate(layer_activations, axis=0)

            if i == num_layers - 1:
                # Skip PCA for the output layer (last layer)
                for subtitle, activation in self.activations.items():
                    if self.reduced_activations.get(subtitle) is None:
                        self.reduced_activations[subtitle] = []

                    # Append the original activation for the output layer
                    self.reduced_activations[subtitle].append(activation[i].cpu().numpy())
            else:
                # Fit PCA model for this layer if pca_components is specified
                if pca_components:
                    if pca_models[i] is None:
                        pca_models[i] = PCA(n_components=pca_components)
                        pca_models[i].fit(concat_activations)
                
                # Transform the activations for each subtitle
                for subtitle, activation in self.activations.items():
                    if self.reduced_activations.get(subtitle) is None:
                        self.reduced_activations[subtitle] = []

                    layer_activation = activation[i].cpu().numpy()

                    if pca_components:
                        # Apply PCA transformation if components are specified
                        reduced_activation = pca_models[i].transform(layer_activation)
                    else:
                        # Otherwise, just append the original activation (no PCA)
                        reduced_activation = layer_activation

                    # Append the reduced activation for the current layer
                    self.reduced_activations[subtitle].append(reduced_activation)

        return self.reduced_activations
    
    def show(self, mean_pooling=False, save=True):
        if self.disable:
            return
        num_layers = len(self.activations[list(self.activations.keys())[0]])

        fig, axs = plt.subplots(len(self.reduced_activations.keys()), 
                                num_layers, 
                                figsize=(num_layers * 6, 
                                         len(self.reduced_activations.keys()) * 4 + 1))
        
        fig.suptitle(self.title, fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        for i in range(num_layers):
            if len(self.reduced_activations.keys()) == 1:
                if mean_pooling:
                    data = self.reduced_activations[list(self.reduced_activations.keys())[0]][i]
                    pooled_data = data.reshape(10, 1000, -1).mean(axis=1)
                    im = axs[i].imshow(pooled_data, aspect='auto')
                else:
                    im = axs[i].imshow(self.reduced_activations[list(self.reduced_activations.keys())[0]][i], aspect='auto')
                axs[i].set_title(list(self.reduced_activations.keys())[0] + f' - Layer {i}', fontsize=12)
                fig.colorbar(im, ax=axs[i])
            else:
                for j, sub_title in enumerate(self.reduced_activations.keys()):
                    if mean_pooling:
                        data = self.reduced_activations[sub_title][i]
                        pooled_data = data.reshape(10, 1000, -1).mean(axis=1)
                        im = axs[j, i].imshow(pooled_data, aspect='auto')
                    else:
                        im = axs[j, i].imshow(self.reduced_activations[sub_title][i], aspect='auto')
                    axs[j, i].set_title(sub_title + f' - Layer {i}', fontsize=12)
                    fig.colorbar(im, ax=axs[j, i])

        self.fig = fig
        plt.show()

        if save:
            self.save()

    def save(self):
        if self.disable:
            return
        self.fig.savefig(self.output_path, bbox_inches='tight', facecolor='w')

# Function to compute stable rank
def compute_stable_rank(nn: SimpleNN):
    stable_ranks = []

    for idx, layer in enumerate(nn.layers):

        weight_matrix = layer.weight.data

        # Compute the Frobenius norm
        fro_norm = torch.norm(weight_matrix, p='fro')
        
        # Compute the spectral norm (largest singular value)
        with torch.no_grad():
            singular_values = torch.linalg.svdvals(weight_matrix)
            sigma_max = singular_values[0]
        
        # Compute the stable rank
        stable_rank = (fro_norm / sigma_max) ** 2

        stable_ranks.append(stable_rank.item())

    return stable_ranks