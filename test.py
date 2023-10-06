import h5py
import pickle
import numpy as np
from typing import Optional, Tuple, List
from collections import defaultdict
from arcle.loaders import ARCLoader, Loader, MiniARCLoader
import gymnasium as gym
import hydra
from hydra.utils import get_original_cwd
import json
from collections import namedtuple
import os


PATH = os.path.dirname(os.path.abspath(__file__))
print(PATH)

arcenv = gym.make('ARCLE/O2ARCv2Env-v0',render_mode=None, data_loader= ARCLoader(), max_grid_size=(30,30), colors = 10, max_episode_steps=None)
minienv = gym.make('ARCLE/O2ARCv2Env-v0',render_mode=None, data_loader= MiniARCLoader(), max_grid_size=(30,30), colors = 10, max_episode_steps=None)

failure_trace = []

def set_env(name):
    for i, aa in enumerate(ARCLoader().data):
        if aa[4]['id'] == name:
            return arcenv
    for i, aa in enumerate(MiniARCLoader().data):
        if aa[4]['id'] == name:
            return minienv

def findbyname(name):
    for i, aa in enumerate(ARCLoader().data):
        if aa[4]['id'] == name:
            return i
    for i, aa in enumerate(MiniARCLoader().data):
        if aa[4]['id'] == name:
            return i

def action_convert(action_entry):
    _, action, data, grid = action_entry
    op = 0
    # print(action, data)
    match action:
        case "CopyFromInput":
            op = 31
        case "ResizeGrid":
            op = 33
        case "ResetGrid":
            op = 32
        case "Submit":
            op = 34
        case "Color":
            op = data[1]
        case "Fill":
            op = data[2]
        case "FlipX":
            op = 27
        case "FlipY":
            op = 26
        case "RotateCW":
            op = 25
        case "RotateCCW":
            op = 24
        case "Move":
            match data[2]:
                case 'U':
                    op = 20
                case 'D':
                    op = 21
                case 'R':
                    op = 22
                case 'L':
                    op = 23
        case "Copy":
            match data[2]:
                case 'Input Grid':
                    op = 28
                case 'Output Grid':
                    op = 29
        case "Paste":
            op = 30
        case "FloodFill":
            op = 10 + data[1]

    return op

@hydra.main(config_path="config", config_name="config.yaml")
def create_features(args):

    traces = []
    traces_info = []

    with open(f"{PATH}/{args.task_config}", "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )

    with open(f"{PATH}/{task_config.traces}", 'rb') as fp:
        traces:List = pickle.load(fp)
    with open(f"{PATH}/{task_config.traces_info}", 'rb') as fp:
        traces_info:List = pickle.load(fp)

    file_no = 8888
    print(f"{PATH}/{task_config.task_paths.format(file_no)}")

create_features()

