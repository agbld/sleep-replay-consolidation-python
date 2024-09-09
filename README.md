# Sleep Replay Consolidation (SRC) - Python

GitHub Repo: https://github.com/agbld/sleep-replay-consolidation-python

This is a Python replica of the original implementation of the Sleep Replay Consolidation (SRC) algorithm, as proposed in [*Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks*](https://doi.org/10.1038/s41467-022-34938-7) (Nature Communications) by Tadros et al., 2022.

The SRC algorithm is designed to mitigate catastrophic forgetting in artificial neural networks by consolidating knowledge across tasks during sleep phases.

The original implementation, [tmtadros/SleepReplayConsolidation](https://github.com/tmtadros/SleepReplayConsolidation.git), was written in MATLAB.

## TODO

- [ ] Figure out the formula for `alpha` and `beta` in the SRC algorithm.
  - [ ] See the *Sleep Replay Consolidation (SRC) algorithm* section under the *Methods* section in the paper.
  - [ ] Also see [*Fast-classifying, high-accuracy spiking deep networks through weight and threshold balancing*](https://ieeexplore.ieee.org/document/7280696) (IJCNN 2015) for the exact scaling algorithm.
- [ ] Implement *model merging* approach and compare it with SRC.
- [ ] Extract the PCA component importance and use it to calculate the model capacity "usage" of each task. Observe how SRC reallocates model capacity during incremental learning.
- [ ] Solve the bias problem.
  - [ ] Currently, the bias is **disabled** in both the official and this Python implementation. However, the paper mentions that the bias is **scaled** during the sleep phase, which means the algorithm should be able to handle the bias as well.
  - [ ] Figure out the current bias scaling algorithm.
  - [ ] If it indeed only does the scaling and won't modify biases during the sleep phase, then the following problems might occur:
    * In a neural network (NN), there are two major types of parameters: weights and biases. Each of them plays a crucial role while learning. The optimizer used in the training phase will try its best to adjust both weights and biases to fit the current task.
    * According to observations, so far, SRC does the "recall" or "memory recovery" by **selectively adjusting the weights**. SRC seeks to find the best compromised weights across all tasks. However, the **bias distribution may shift across tasks** as well.
    * If the target tasks are somehow **"bias-sensitive"**, then SRC might **suffer from the "outdated" bias**, resulting in catastrophic forgetting again.

## Experiments

Following are the experiments conducted solely on this Python implementation.

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