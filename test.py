import pickle
import dill
import torch
import torch.optim as O
import higher
from itertools import count
from train import get_env, get_opts_and_lrs
from losses import policy_loss_on_batch, vf_loss_on_batch
import hydra
import json
import logging
from collections import namedtuple
import os


LOG = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.abspath(__file__))

@hydra.main(config_path="config", config_name="config.yaml")
def run(args):
    with open(f"{PATH}/{args.task_config}", "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )
    env = get_env(args, task_config)

    policy = torch.load(f"{PATH}/{task_config.model_paths}/{task_config.model_policy}")
    vf = torch.load(f"{PATH}/{task_config.model_paths}/{task_config.model_policy}")

    with open(f"{PATH}/{task_config.eval_task_paths}/{task_config.eval_task}", "r") as f:
        eval_task = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )

    # num_train_tasks = len(eval_task.train)
    # for i, train_task in enumerate(eval_task.train):
    #     print("TASK", i)
    #     print("INPUT:", eval_task.train[i].input)
    #     print("OUTPUT:", eval_task.train[i].output)

    # print()
    # for i, test_task in enumerate(eval_task.test):
    #     print("TASK", i)
    #     print("INPUT:", eval_task.test[i].input)
    #     print("OUTPUT:", eval_task.test[i].output)

    policy_opt, vf_opt, policy_lrs, vf_lrs = get_opts_and_lrs(args, policy, vf)

    for train_step_idx in count(start=1):
        if train_step_idx % args.rollout_interval == 0:
            LOG.info(f"Train step {train_step_idx}")

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

                meta_vf_loss = vf_loss_on_batch(f_vf, outer_batch)
                total_vf_loss = meta_vf_loss / len(task_config.train_tasks)
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
                    f_policy, adapted_vf, outer_batch, args.advantage_head_coef
                )

                (meta_policy_loss / len(task_config.train_tasks)).backward()

                # Sample adapted policy trajectory
                # if train_step_idx % args.rollout_interval == 0:
                #     adapted_trajectory, adapted_reward, success = rollout_policy(f_policy, env)
                #     LOG.info(f"Task {train_task_idx} reward: {adapted_reward}")

        # Update the policy/value function
        policy_opt.step()
        policy_opt.zero_grad()
        vf_opt.step()
        vf_opt.zero_grad()

        if train_step_idx % args.rollout_interval == 0:
            for test_task_idx, task_buffers in test_task_buffers:
                env.set_task_idx(test_task_idx)
                adapted_trajectory, adapted_reward, success = rollout_policy(policy, env, True)
                LOG.info(f"Task {test_task_idx} reward: {adapted_reward}")
            
            
        if train_step_idx % args.model_save_interval == 0:
            train_s, train_e = train_task_buffers[0][0], train_task_buffers[-1][0] + 1
            test_s, test_e = test_task_buffers[0][0], test_task_buffers[-1][0] + 1

            torch.save(policy, f"{PATH}/{task_config.model_path}/policy_steps_{train_step_idx}_train_{train_s}_{train_e}_test_{test_s}_{test_e}.pt", pickle_module=dill)
            torch.save(vf, f"{PATH}/{task_config.model_path}/vf_steps_{train_step_idx}_train_{train_s}_{train_e}_test_{test_s}_{test_e}.pt", pickle_module=dill)

if __name__ == "__main__":
    run()

