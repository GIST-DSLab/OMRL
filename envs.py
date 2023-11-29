import numpy as np
from typing import Optional, Tuple, List
from copy import deepcopy
import gymnasium as gym
from arcle.loaders import ARCLoader, Loader, MiniARCLoader

class ArcEnv(gym.Env):
    def __init__(self, traces: List, traces_info: List, include_goal: bool = False):
        self.include_goal = include_goal
        super(ArcEnv, self).__init__()
        self.arcloader = ARCLoader()
        self.arcloader_eval = ARCLoader(train=False)
        self.miniarcloader = MiniARCLoader()
        self.arcenv = gym.make('ARCLE/O2ARCv2Env-v0', render_mode=None, data_loader=self.arcloader, max_grid_size=(30,30), colors=10, max_episode_steps=None)
        self.arcenv_eval = gym.make('ARCLE/O2ARCv2Env-v0', render_mode=None, data_loader=self.arcloader_eval, max_grid_size=(30,30), colors=10, max_episode_steps=None)
        self.miniarcenv = gym.make('ARCLE/O2ARCv2Env-v0', render_mode=None, data_loader=self.miniarcloader, max_grid_size=(30,30), colors=10, max_episode_steps=None)
        self.env = self.arcenv
        self.traces = traces
        self.traces_info = traces_info
        self._max_episode_steps = 200
        self.idx = 1
        self._task = None
        
    def _get_obs(self):
        if self.include_goal:
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[self.idx] = 1.0 # one_hot = [0, 0, ..., 1, 0, ... 0] (one_hot[idx] = 1)
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot]) # obs += one_hot
        else:
            obs = super()._get_obs()
        return obs
    
    def get_idx(self):
        return self.idx

    def findbyname(self, name):
        for i, aa in enumerate(self.arcloader.data):
            if aa[4]['id'] == name:
                self.env = self.arcenv
                return i
        for i, aa in enumerate(self.arcloader_eval.data):
            if aa[4]['id'] == name:
                self.env = self.arcenv_eval
                return i
        for i, aa in enumerate(self.miniarcloader.data):
            if aa[4]['id'] == name:
                self.env = self.miniarcenv
                return i

    def covert_action_info(self, action_entry):
        _, action, data, grid = action_entry
        sel = np.zeros((30,30), dtype=np.bool_)
        op = 0
        if action == "CopyFromInput":
            op = 31
            bbox = [[0, 0], [0, 0]]
        elif action == "ResizeGrid":
            op = 33
            h, w = data[0]
            # sel[:h,:w] = 1
            bbox = [[0, 0], [h, w]]
        elif action == "ResetGrid":
            op = 32
            bbox = [[0, 0], [0, 0]]
        elif action == "Submit":
            op = 34
            bbox = [[0, 0], [0, 0]]
        elif action == "Color":
            h, w = data[0]
            op = data[1]
            # sel[h,w] = 1
            bbox = [[h, w], [h, w]]
        elif action == "Fill":
            h0, w0 = data[0]
            h1, w1 = data[1]
            op = data[2]
            # sel[h0:h1+1 , w0:w1+1] = 1
            bbox = [[h0, w0], [h1, w1]]
        elif action == "FlipX":
            h0, w0 = data[0]
            h1, w1 = data[1]
            op = 27
            # sel[h0:h1+1, w0:w1+1] = 1
            bbox = [[h0, w0], [h1, w1]]
        elif action == "FlipY":
            h0, w0 = data[0]
            h1, w1 = data[1]
            op = 26
            # sel[h0:h1+1, w0:w1+1] = 1
            bbox = [[h0, w0], [h1, w1]]
        elif action == "RotateCW":
            h0, w0 = data[0]
            h1, w1 = data[1]
            op = 25
            # sel[h0:h1+1, w0:w1+1] = 1
            bbox = [[h0, w0], [h1, w1]]
        elif action == "RotateCCW":
            h0, w0 = data[0]
            h1, w1 = data[1]
            op = 24
            # sel[h0:h1+1, w0:w1+1] = 1
            bbox = [[h0, w0], [h1, w1]]
        elif action == "Move":
            h0, w0 = data[0]
            h1, w1 = data[1]
            if data[2] == 'U':
                op = 20
            elif data[2] == 'D':
                op = 21
            elif data[2] == 'R':
                op = 22
            elif data[2] == 'L':
                op = 23
            # sel[h0:h1+1, w0:w1+1] = 1
            bbox = [[h0, w0], [h1, w1]]
        elif action == "Copy":
            h0, w0 = data[0]
            h1, w1 = data[1]
            
            if data[2] == 'Input Grid':
                op = 28
            elif data[2] == 'Output Grid':
                op = 29
            # sel[h0:h1+1, w0:w1+1] = 1
            bbox = [[h0, w0], [h1, w1]]
        elif action == "Paste":
            h, w = data[0]
            op = 30
            # sel[h,w] = 1
            bbox = [[h, w], [h, w]]
        elif action == "FloodFill":
            h, w = data[0]
            op = 10 + data[1]
            # sel[h,w] = 1
            bbox = [[h, w], [h, w]]
        return op, bbox

    def set_task(self, task):
        self._task = task
        # self._goal_dir = self._task['direction']
        state = self.env.reset(options= {'adaptation':False, 'prob_index':self.findbyname(self.traces_info[self.idx][0]), 'subprob_index': self.traces_info[self.idx][1]})

    def set_task_idx(self, idx):
        self.idx = idx
        self.findbyname(self.traces_info[self.idx][0])
        self.set_task(self.traces[self.idx])

    def set_task_test(self, task_name):
        state = self.env.reset(options= {'adaptation':False, 'prob_index':self.findbyname(task_name), 'subprob_index': 0})
    