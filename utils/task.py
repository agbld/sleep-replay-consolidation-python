import numpy as np

def create_class_task(X, Y, task_size, num_tasks=None):
    Xs = []
    Ys = []

    unique_classes = np.unique(Y)
    task_id = 0

    # Assign each pair of classes to a task
    for i in range(0, len(unique_classes), task_size):
        class_group = unique_classes[i:i+task_size]
        indices = np.isin(Y, class_group)
        Xs.append(X[indices])
        Ys.append(Y[indices])

        task_id += 1

    # Assign tasks with id > num_tasks to -1
    if num_tasks is not None:
        Xs = Xs[:num_tasks]
        Ys = Ys[:num_tasks]

    return Xs, Ys

import numpy as np

def create_permutation_task(X, Y, num_tasks, perms: list = None):
    Xs = []
    Ys = []
    
    num_features = X.shape[1]  # Number of pixels in each image
    
    used_perms = []
    for _ in range(num_tasks):
        if perms is not None:
            perm = perms[_]
        else:
            # Generate a random permutation of pixel indices
            perm = np.random.permutation(num_features)
        used_perms.append(perm)

        # Apply the permutation to all images
        X_permuted = X[:, perm]
        Xs.append(X_permuted)
        Ys.append(Y.copy())  # Labels remain the same for each task
    
    return Xs, Ys, used_perms
