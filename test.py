 import pickle
import dill
import torch
import torch.optim as O
import higher
from itertools import count
from train import get_env, get_opts_and_lrs, rollout_policy
from losses import policy_loss_on_batch, vf_loss_on_batch
import json
import logging
from collections import namedtuple
import os
import glob
import numpy as np
import yaml
from omegaconf import OmegaConf
import wandb

# LOG = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.abspath(__file__))
wandb.login()

def run():
    with open(f"{PATH}/config/config.yaml", "r") as y:
        args = OmegaConf.create(yaml.load(y, Loader=yaml.FullLoader))

    total_reward = {}
    acc, tot = 0, 0

    wandb.init(project='macaw-min', config=dict(args))

    '''<<<< LOAD ARGUMENTS >>>>'''
    with open(f"{PATH}/{args.task_config}", "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )

    '''<<<< GET ARC ENVS >>>>'''
    env = get_env(args, task_config)

    for j, filename in enumerate(glob.glob(f"{PATH}/{task_config.eval_task_paths}/*.json")):
        tot += 1

        '''<<<< LOAD TRAINED MODEL >>>>'''
        policy = torch.load(f"{PATH}/{task_config.model_paths}/{task_config.model_policy}")
        vf = torch.load(f"{PATH}/{task_config.model_paths}/{task_config.model_policy}")
        policy_opt, vf_opt, policy_lrs, vf_lrs = get_opts_and_lrs(args, policy, vf)
        # print(policy, vf)

        '''<<<< LOAD EVALUATION TASKS >>>>'''
        with open(filename, "r") as f:
            eval_task = json.load(
                f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
            )

        for eval_train in eval_task.train:
            eval_train_input = np.array(eval_train.input)
            eval_train_output = np.array(eval_train.output)
            # print(filename.split('/')[-1][:-5])
        
        for eval_test in eval_task.test:
            eval_test_input = np.array(eval_test.input)
            eval_test_output = np.array(eval_test.output)
            # print(eval_test_input)
            # print(eval_test_output)
        
        test_task_name = filename.split('/')[-1][:-5]
        # print(test_task_name)
        env.set_task_test(test_task_name)

        '''<<<< CREATE BUFFER FOR EVALUATION TASKS >>>>'''
        '''<<<< FINETUNE MODEL WITH EVALUATION TRAIN INPUT/OUTPUT PAIRS >>>>'''
        # for train_step_idx in range(1, task_config.finetune_steps + 1):
        #     for train_task_idx, train_task in enumerate(eval_task.train):
        #         env.set_task_idx(train_task_idx)

        #         inner_batch = task_buffers.sample(
        #             args.inner_batch_size, return_dict=True, device=args.device
        #         )
        #         outer_batch = task_buffers.sample(
        #             args.outer_batch_size, return_dict=True, device=args.device
        #         )

        #         # Adapt value function
        #         opt = O.SGD([{"params": p, "lr": None} for p in vf.parameters()])
        #         with higher.innerloop_ctx(
        #             vf, opt, override={"lr": vf_lrs}, copy_initial_weights=False
        #         ) as (f_vf, diff_value_opt):
        #             loss = vf_loss_on_batch(f_vf, inner_batch, inner=True)
        #             diff_value_opt.step(loss)

        #             meta_vf_loss = vf_loss_on_batch(f_vf, outer_batch)
        #             total_vf_loss = meta_vf_loss / len(task_config.train_tasks)
        #             total_vf_loss.backward()

        #         # Adapt policy using adapted value function
        #         adapted_vf = f_vf
        #         opt = O.SGD([{"params": p, "lr": None} for p in policy.parameters()])
        #         with higher.innerloop_ctx(
        #             policy, opt, override={"lr": policy_lrs}, copy_initial_weights=False
        #         ) as (f_policy, diff_policy_opt):
        #             loss = policy_loss_on_batch(
        #                 f_policy,
        #                 adapted_vf,
        #                 inner_batch,
        #                 args.advantage_head_coef,
        #                 inner=True,
        #             )

        #             diff_policy_opt.step(loss)
        #             meta_policy_loss = policy_loss_on_batch(
        #                 f_policy, adapted_vf, outer_batch, args.advantage_head_coef
        #             )

        #             (meta_policy_loss / len(task_config.train_tasks)).backward()

        #             # Sample adapted policy trajectory
        #             # if train_step_idx % args.rollout_interval == 0:
        #             #     adapted_trajectory, adapted_reward, success = rollout_policy(f_policy, env)
        #             #     LOG.info(f"Task {train_task_idx} reward: {adapted_reward}")

        #     # Update the policy/value function
        #     policy_opt.step()
        #     policy_opt.zero_grad()
        #     vf_opt.step()
        #     vf_opt.zero_grad()
            
        '''<<<< PREDICT OUTPUT FOR EVALUATION TEST INPUT >>>>'''
        adapted_trajectory, adapted_reward, success = rollout_policy(policy, env, True)
        submit_count = 0
        for i in range(len(adapted_trajectory) - 1, -1, -1):
            # if adapted_trajectory[i].action != 34:
            #     continue # PROBLEM: current model doesn't submit

            submit_count += 1
            predicted_test_output = adapted_trajectory[i].state['grid']

            h, w = eval_test_output.shape
            predicted_test_output = predicted_test_output[:h, :w]

            if np.array_equal(eval_test_output, predicted_test_output):
                acc += 1
                break
            elif submit_count == 3:
                break

        # LOG.info(f"Task {test_task_name} reward: {adapted_reward}")
        total_reward[test_task_name] = adapted_reward


    sum_adapted_reward = sum(list(total_reward.values()))
    acc /= tot
    wandb.log({"sum_adapted_reward": sum_adapted_reward, "accuracy": acc})

    return sum_adapted_reward, acc

if __name__ == "__main__":
    result = run()
    print(result)
