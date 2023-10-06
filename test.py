import pickle
import torch
import numpy as np
from typing import Optional, Tuple, List
from collections import defaultdict
from arcle.loaders import ARCLoader, Loader, MiniARCLoader

import gymnasium as gym
import hydra
import json
from collections import namedtuple
import os


PATH = os.path.dirname(os.path.abspath(__file__))

@hydra.main(config_path="config", config_name="config.yaml")
def run(args):
    with open(f"{PATH}/{args.task_config}", "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )

    policy = torch.load(f"{PATH}/{task_config.model_paths}/{task_config.model_policy}")
    vf = torch.load(f"{PATH}/{task_config.model_paths}/{task_config.model_policy}")

    with open(f"{PATH}/{task_config.eval_task_paths}/{task_config.eval_task}", "r") as f:
        eval_task = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )

    num_train_tasks = len(eval_task.train)
    for i, train_task in enumerate(eval_task.train):
        print("TASK", i)
        print("INPUT:", eval_task.train[i].input)
        print("OUTPUT:", eval_task.train[i].output)

    print()
    for i, test_task in enumerate(eval_task.test):
        print("TASK", i)
        print("INPUT:", eval_task.test[i].input)
        print("OUTPUT:", eval_task.test[i].output)

if __name__ == "__main__":
    run()

