# Minimal implementation of MACAW (ICML 2021)

This repo contains a simplified implementation of the MACAW algorithm (full code [here](https://github.com/eric-mitchell/macaw)) for easier inspection and extension.

Run the code like below.

`python create_buffers.py` (optional: if you want to make buffers with offline data).

`python train.py` (optional: if you want to train meta-RL models).

`python test.py`.

## Overview

This code trains MACAW on the simple Cheetah-Direction problem, which has only two tasks (forwards and backwards). `impl.py` contains example of loading the offline data (`build_networks_and_buffers`) and performing meta-training (loop in `run.py`). `losses.py` contains the MACAW loss functions for adaptation the value function and policy. `utils.py` contains the replay buffer implementation that loads the offline data.

# Citation

If our code or research was useful for your own work, you can cite us with the following attribution:

    @InProceedings{mitchell2021offline,
        title = {Offline Meta-Reinforcement Learning with Advantage Weighting},
        author = {Mitchell, Eric and Rafailov, Rafael and Peng, Xue Bin and Levine, Sergey and Finn, Chelsea},
        booktitle = {Proceedings of the 38th International Conference on Machine Learning},
        year = {2021}
    }
