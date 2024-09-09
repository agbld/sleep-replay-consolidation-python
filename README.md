# Sleep Replay Consolidation (SRC) - Python

GitHub Repo: https://github.com/agbld/sleep-replay-consolidation-python

This is a replicate of the original implementation of the Sleep Replay Consolidation (SRC) algorithm in Python. The original implementation [tmtadros/SleepReplayConsolidation](https://github.com/tmtadros/SleepReplayConsolidation.git) was in MATLAB and was used in the following paper:
* Tadros, T., Krishnan, G. P., Ramyaa, R., & Bazhenov, M. (2022a). *Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks*. Nature Communications, 13(1). https://doi.org/10.1038/s41467-022-34938-7

## Experiments

### MNIST

The `mnist_exp.py` script try to replicate the original experiment ([see here!](https://github.com/tmtadros/SleepReplayConsolidation/blob/main/MNIST/run_class_task_summary.m)) on the MNIST dataset. Using the SRC algorithm ([see here!](https://github.com/tmtadros/SleepReplayConsolidation/blob/main/sleep/sleepnn_old.m)) that originally implemented in MATLAB.

#### Accuracy

![image](https://hackmd.io/_uploads/rk4RN-hh0.png)

#### Neuron Activity Visualization

The following visualizations show the activations for each model across different tasks in sequence.

Each row of subplots contains three subplots, each representing a layer, with Layer 0 on the left through Layer 2 (the output layer) on the right.

In each subplot, the **x-axis** represents different ten **"neuron clusters"**. PCA is applied to reduce the dimensionality of each layer's activations from 4000 to 10. The **PCA projection matrices** are **aligned under the same layer in the same task**, but **differ across tasks** due to the nature of PCA. Note that the **last layer (Layer 2)** shows raw activation values **without PCA**, as it already consists of 10 dimensions. The **y-axis** represents the **input indices**, which are ordered by class index from top to bottom as class 0, class 1, ..., class 9. Each class contains 1000 samples.

##### SRC - Before, After, and Difference

![](./png/layer_activations_task_0_before_after_src.png)
![](./png/layer_activations_task_1_before_after_src.png)
![](./png/layer_activations_task_2_before_after_src.png)
![](./png/layer_activations_task_3_before_after_src.png)
![](./png/layer_activations_task_4_before_after_src.png)

##### Sequential Learning (Lower Bound)

![](./png/layer_activations_task_0_sequential.png)
![](./png/layer_activations_task_1_sequential.png)
![](./png/layer_activations_task_2_sequential.png)
![](./png/layer_activations_task_3_sequential.png)
![](./png/layer_activations_task_4_sequential.png)

##### Parallel Learning (Upper Bound)

![](./png/layer_activations_parallel.png)

<!-- ### Ideas
* Instead of using a **fixed amount** for increasing or decreasing, a slight **"step-back"** to the previous model state (before SRC) might be better. (?)
    * According to the dynamics of visualized activations, the SRC algorithm **does not** seem to actually "recover" lost memories. 
    * Instead, it **selectively "cancels" the model's ability** on the current task and makes the model **"less focused" on the current task**. 
    * SRC causes the model to be **less confident** in everything, especially newly learned tasks. -->