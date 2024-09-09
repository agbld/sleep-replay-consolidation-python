import numpy as np

def create_class_task(y, task_size=2):
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

    return tasks