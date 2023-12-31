from nn import MLP
from macaw_dt import Config, DT
from utils import ReplayBuffer
import yaml
from omegaconf import OmegaConf
import json
from collections import namedtuple
import pickle
import torch
import torch.optim as O
from typing import List
import higher
from itertools import count
from utils import Experience
from losses import policy_loss_on_batch, vf_loss_on_batch
from envs import ArcEnv
import numpy as np
import dill
import os
import wandb

torch.autograd.set_detect_anomaly(True)

PATH = os.path.dirname(os.path.abspath(__file__))
wandb.login()

def rollout_policy(policy: MLP, env, render: bool = False) -> List[Experience]:
    trajectory = []
    idx = env.get_idx()
    state = env.env.reset(options= {'adaptation':False, 'prob_index':env.findbyname(env.traces_info[idx][0]), 'subprob_index': env.traces_info[idx][1]})
    if render:
        env.env.render()
    done = False
    total_reward = 0
    episode_t = 0
    success = False
    policy.eval()
    current_device = list(policy.parameters())[-1].device
    while not done:
        with torch.no_grad():
            # some state is like ({'selected':array(x, y), 'grid':array(x, y), 'grid_dim':(x, y), 'clip':array(x, y), 'clip_dim':(x, y), ...}, {'steps': x, ...})
            if len(state) == 2:
                state = state[0]

            np_action = policy(torch.tensor(state['grid'].reshape(1, -1)).to(current_device).float()).squeeze().cpu().numpy()
            action = dict()
            try:
                action['operation'] = int(np.interp(np_action[0], (-1, 1), (1, 34)))
                
                x0, y0, h, w = int(np.interp(np_action[1:], (-1, 1), (0, 30)))
                action['selection'] = np.zeros((30,30), dtype=np.bool_)
                action['selection'][x0:min(30, x0+h), y0:min(30, y0+w)] = True
            except:
                action['operation'] = 1

                x0, y0, h, w = (0, 0, 30, 30)
                action['selection'] = np.zeros((30,30), dtype=np.bool_)
                action['selection'][x0:min(30, x0+h), y0:min(30, y0+w)] = True

        next_state, reward, done, _, info_dict = env.env.step(action)

        if "success" in info_dict and info_dict["success"]:
            success = True

        if render:
            env.env.render()
        trajectory.append(Experience(state, np_action, next_state, reward, done))
        state = next_state
        total_reward += reward
        episode_t += 1
        if episode_t >= env._max_episode_steps or done:
            break

    return trajectory, total_reward, success


def build_networks_and_buffers(args, env, task_config, is_MLP = False):
    obs_dim = 900
    action_dim = 5

    policy_head = [32, 1] if args.advantage_head_coef is not None else None

    if is_MLP:
        policy = MLP(
            [obs_dim] + [args.net_width] * args.net_depth + [action_dim],
            final_activation=torch.tanh,
            extra_head_layers=policy_head,
            w_linear=args.weight_transform,
        ).to(args.device)

        vf = MLP(
            [obs_dim] + [args.net_width] * args.net_depth + [1],
            w_linear=args.weight_transform,
        ).to(args.device)
    else:
        policy_config = Config()
        policy_config.loss_dim = action_dim
        policy = DT(policy_config).to(args.device)

        vf_config = Config()
        policy_config.loss_dim = 1
        vf = DT(vf_config).to(args.device)

    s, e = map(int, task_config.train_tasks)
    train_buffer_paths = [
        (idx, f"{PATH}/{task_config.train_buffer_paths.format(idx)}") for idx in range(s, e)
    ]

    train_buffers = [
        (idx, ReplayBuffer(
            args.inner_buffer_size,
            obs_dim,
            action_dim,
            discount_factor=0.99,
            immutable=True,
            load_from=train_buffer,
        ))
        for idx, train_buffer in train_buffer_paths
    ]
    
    s, e = map(int, task_config.test_tasks)
    test_buffer_paths = [
        (idx, f"{PATH}/{task_config.test_buffer_paths.format(idx)}") for idx in range(s, e)
    ]

    test_buffers = [
        (idx, ReplayBuffer(
            args.inner_buffer_size,
            obs_dim,
            action_dim,
            discount_factor=0.99,
            immutable=True,
            load_from=test_buffer,
        ))
        for idx, test_buffer in test_buffer_paths
    ]

    return policy, vf, train_buffers, test_buffers

def get_env(args, task_config):
    traces = []
    traces_info = []
    # for task_idx in range(task_config.total_tasks):
    #     with open(task_config.task_paths.format(task_idx), "rb") as f:
    #         task_info = pickle.load(f)
    #         assert len(task_info) == 1, f"Unexpected task info: {task_info}"
    #         tasks.append(task_info[0])
    # if args.advantage_head_coef == 0:
    #     args.advantage_head_coef = None  
    with open(f"{PATH}/{task_config.traces}", 'rb') as fp:
        traces:List = pickle.load(fp)
    with open(f"{PATH}/{task_config.traces_info}", 'rb') as fp:
        traces_info:List = pickle.load(fp)

    return ArcEnv(traces=traces, traces_info=traces_info, include_goal=True)


def get_opts_and_lrs(args, policy, vf):
    policy_opt = O.Adam(policy.parameters(), lr=args.outer_policy_lr)
    vf_opt = O.Adam(vf.parameters(), lr=args.outer_value_lr)
    policy_lrs = [
        torch.nn.Parameter(torch.tensor(args.inner_policy_lr).to(args.device))
        for p in policy.parameters()
    ]
    vf_lrs = [
        torch.nn.Parameter(torch.tensor(args.inner_value_lr).to(args.device))
        for p in vf.parameters()
    ]

    return policy_opt, vf_opt, policy_lrs, vf_lrs


def train():
    with open(f"{PATH}/config/config.yaml", "r") as y:
        args = OmegaConf.create(yaml.load(y, Loader=yaml.FullLoader))

    wandb.init(project='macaw-min', config=dict(args))

    with open(f"{PATH}/{args.task_config}", "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )
    
    env = get_env(args, task_config)

    policy, vf, train_task_buffers, test_task_buffers = build_networks_and_buffers(args, env, task_config)
    policy_opt, vf_opt, policy_lrs, vf_lrs = get_opts_and_lrs(args, policy, vf)

    for train_step_idx in count(start=1):
        for train_task_idx, task_buffers in train_task_buffers:
            env.set_task_idx(train_task_idx)

            inner_batch = task_buffers.sample(
                args.inner_batch_size, return_dict=True, device=args.device
            )
            outer_batch = task_buffers.sample(
                args.outer_batch_size, return_dict=True, device=args.device
            )

            # Adapt value function
            opt = O.SGD([{"params": p, "lr": None} for p in vf.parameters()])
            with higher.innerloop_ctx(
                vf, opt, override={"lr": vf_lrs}, copy_initial_weights=False
            ) as (f_vf, diff_value_opt):
                loss = vf_loss_on_batch(f_vf, inner_batch, inner=True)
                diff_value_opt.step(loss)

                s, e = map(int, task_config.train_tasks)
                meta_vf_loss = vf_loss_on_batch(f_vf, outer_batch)
                total_vf_loss = meta_vf_loss / (e - s)
                total_vf_loss.backward()

            # Adapt policy using adapted value function
            adapted_vf = f_vf
            opt = O.SGD([{"params": p, "lr": None} for p in policy.parameters()])
            with higher.innerloop_ctx(
                policy, opt, override={"lr": policy_lrs}, copy_initial_weights=False
            ) as (f_policy, diff_policy_opt):
                loss = policy_loss_on_batch(
                    f_policy,
                    adapted_vf,
                    inner_batch,
                    args.advantage_head_coef,
                    inner=True,
                )

                diff_policy_opt.step(loss)
                meta_policy_loss = policy_loss_on_batch(
                    f_policy, 
                    adapted_vf, 
                    outer_batch, 
                    args.advantage_head_coef,
                    inner=False
                )

                s, e = map(int, task_config.train_tasks)
                try:
                    (meta_policy_loss / (e - s)).backward()
                except:
                    import pdb;pdb.set_trace()

        # Update the policy/value function
        max_norm = 5
        policy_opt.zero_grad()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)
        policy_opt.step()
        vf_opt.zero_grad()
        torch.nn.utils.clip_grad_norm_(vf.parameters(), max_norm)
        vf_opt.step()

        if train_step_idx % args.rollout_interval == 0:
            sum_adapted_reward = 0
            for test_task_idx, task_buffers in test_task_buffers:
                env.set_task_idx(test_task_idx)
                adapted_trajectory, adapted_reward, success = rollout_policy(policy, env, True)
                sum_adapted_reward += adapted_reward
            wandb.log({"sum_adapted_reward": sum_adapted_reward}, step=train_step_idx)
            
            
        if train_step_idx % args.model_save_interval == 0:
            train_s, train_e = train_task_buffers[0][0], train_task_buffers[-1][0] + 1
            test_s, test_e = test_task_buffers[0][0], test_task_buffers[-1][0] + 1

            torch.save(policy, f"{PATH}/{task_config.model_paths}/policy_steps_{train_step_idx}_train_{train_s}_{train_e}_test_{test_s}_{test_e}.pt", pickle_module=dill)
            torch.save(vf, f"{PATH}/{task_config.model_paths}/vf_steps_{train_step_idx}_train_{train_s}_{train_e}_test_{test_s}_{test_e}.pt", pickle_module=dill)

if __name__ == "__main__":
    train()