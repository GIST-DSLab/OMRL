import h5py
import pickle
import numpy as np
from typing import List
from collections import defaultdict
from envs import ArcEnv
import hydra
import json
from collections import namedtuple
import os


PATH = os.path.dirname(os.path.abspath(__file__))

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

    env = ArcEnv(traces=traces, traces_info=traces_info, include_goal=True)

    task_dict = defaultdict(list)
    for idx, (trace, info) in enumerate(zip(traces, traces_info)):
        print(idx)
        name, subtask, isGoal = info
        prob_index = env.findbyname(name)
        obs_init, _ = env.env.reset(options={'adaptation':False, 'prob_index': prob_index, 'subprob_index': subtask})
        obs_answer = np.zeros(shape=(30, 30))
        for x in range(env.env.answer.shape[0]):
            for y in range(env.env.answer.shape[1]):
                obs_answer = env.env.answer[x][y]
        task_dict[f"{name}_{subtask}"].append((idx, obs_init, obs_answer))


    discount_factor = 0.99
    file_no = 0
    for task in task_dict.keys():
        id_list, obs_init_list, obs_answer_list = zip(*task_dict[task])
        task_no, subtask_no = task.split('_')

        cnt = sum([len(traces[id]) - 1 for id in id_list])
        obs = np.zeros(shape=(cnt, 30, 30))
        next_obs = np.zeros(shape=(cnt, 30, 30))
        terminal_obs = np.zeros(shape=(cnt, 1), dtype=bool)
        terminals = np.zeros(shape=(cnt, 30, 30))
        actions_num = np.zeros(shape=(cnt, 1))
        actions_bbox = np.zeros(shape=(cnt, 4))
        # actions_clip = np.zeros(shape=(cnt, 4))
        rewards = np.zeros(shape=(cnt, 1))
        mc_rewards = np.zeros(shape=(cnt, 1))
        terminal_discounts = np.zeros(shape=(cnt, 1))

        cnt = 0
        for id, obs_init, obs_answer in zip(id_list, obs_init_list, obs_answer_list):
            # input image 
            obs_first = np.zeros(shape=(30, 30))
            for x in range(obs_init['grid_dim'][0]):
                for y in range(obs_init['grid_dim'][1]):
                    obs_first[x][y] = obs_init['grid'][x][y]

            # obs_answer로 대체
            # output image: obs_terminal이 그냥 현재 trace의 마지막 state로 되어있음(정답이 아닐 수 있음)
            # obs_terminal = np.zeros(shape=(30, 30))
            # for x in range(traces[id][-1][-1].shape[0]):
            #     for y in range(traces[id][-1][-1].shape[1]):
            #         obs_terminal[x][y] = traces[id][-1][-1][x][y]
            
            obs_after = obs_answer.copy()
            # obs_after = traces[id][-2][-1].copy()
            print("!!!!", obs_after.shape)

            for i in range(len(traces[id]) - 2, -1, -1): # skip commit actions
                if i == 0:
                    obs_before = obs_first.copy()
                else:
                    obs_before = np.zeros(shape=(30, 30))
                    for x in range(traces[id][i-1][-1].shape[0]):
                        for y in range(traces[id][i-1][-1].shape[1]):
                            obs_before[x][y] = traces[id][i-1][-1][x][y]

                obs[cnt] = obs_before.copy()
                next_obs[cnt] = obs_after.copy()
                terminal_obs[cnt] = obs_answer.copy()
                actions_num[cnt], actions_bbox[cnt] =  env.covert_action_info(traces[id][i])
                import pdb; pdb.set_trace()

                isTerminal = True
                for x in range(30):
                    for y in range(30):
                        if obs[cnt][x][y] != obs_answer[x][y]:
                            isTerminal = False
                            break
                    if not isTerminal:
                        break

                if isTerminal:
                    terminals[cnt] = True
                    rewards[cnt] = 1 # sparse rewards 
                    mc_rewards[cnt] = rewards[cnt]
                    terminal_discounts[cnt] = discount_factor
                else:
                    mc_rewards[cnt] = rewards[cnt] + discount_factor * mc_rewards[cnt - 1]
                    terminal_discounts[cnt] = discount_factor * terminal_discounts[cnt - 1]

                obs_after = obs_before.copy()
                cnt += 1
        
        with open(f"{PATH}/{task_config.task_paths.format(file_no)}", 'wb') as f:
            li = [{}]
            li[0]['task_no'] = task_no
            li[0]['subtask_no'] = subtask_no
            pickle.dump(li, f, pickle.HIGHEST_PROTOCOL)

        with h5py.File(f"{PATH}/{task_config.train_buffer_paths.format(file_no)}", 'w') as f:
            f.create_dataset('obs', data=obs.reshape(cnt, 900), maxshape = (cnt, 900))
            f.create_dataset('next_obs', data=next_obs.reshape(cnt, 900), maxshape = (cnt, 900))
            f.create_dataset('terminal_obs', data=terminal_obs.reshape(cnt, 900), maxshape = (cnt, 900))
            f.create_dataset('terminals', data=terminals, maxshape = (cnt, 1))
            f.create_dataset('actions', data=actions, maxshape = (cnt, 1))
            
            f.create_dataset('rewards', data=rewards, maxshape = (cnt, 1))
            f.create_dataset('mc_rewards', data=mc_rewards, maxshape = (cnt, 1))
            f.create_dataset('discount_factor', data=discount_factor, maxshape = ())
            f.create_dataset('terminal_discounts', data=terminal_discounts, maxshape = (cnt, 1))
        
        file_no += 1

if __name__ == "__main__":
    create_features()