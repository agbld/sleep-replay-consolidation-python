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