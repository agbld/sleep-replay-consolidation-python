# Sleep Replay Consolidation (SRC) - Python

This is a replicate of the original implementation of the Sleep Replay Consolidation (SRC) algorithm in Python. The original implementation [tmtadros/SleepReplayConsolidation](https://github.com/tmtadros/SleepReplayConsolidation.git) was in MATLAB and was used in the following paper:
* Tadros, T., Krishnan, G. P., Ramyaa, R., & Bazhenov, M. (2022a). *Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks*. Nature Communications, 13(1). https://doi.org/10.1038/s41467-022-34938-7

## Experiments

### MNIST

The `mnist_exp.py` script try to replicate the original experiment ([see here!](https://github.com/tmtadros/SleepReplayConsolidation/blob/main/MNIST/run_class_task_summary.m)) on the MNIST dataset. Using the SRC algorithm ([see here!](https://github.com/tmtadros/SleepReplayConsolidation/blob/main/sleep/sleepnn_old.m)) that originally implemented in MATLAB.