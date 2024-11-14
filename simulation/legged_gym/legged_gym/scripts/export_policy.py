from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import copy

from legged_gym.envs import *
from legged_gym.gym_utils import get_args, task_registry, Logger
from legged_gym.gym_utils.helpers import get_load_path, class_to_dict
from datetime import datetime

import torch
from rsl_rl.modules import ActorCriticExplicit

class Export_explicit(torch.nn.Module):
    def __init__(self, actor, long_history, state_estimator, num_short_obs, in_channels, num_proprio_obs):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.long_history = copy.deepcopy(long_history).cpu()
        self.state_estimator = copy.deepcopy(state_estimator).cpu()
        self.num_short_obs = num_short_obs
        self.in_channels = in_channels
        self.num_proprio_obs = num_proprio_obs

    def forward(self, observations):
        short_history = observations[...,-self.num_short_obs:]
        es_vel = self.state_estimator(short_history)
        compressed_long_history = self.long_history(observations.view(-1, self.in_channels, self.num_proprio_obs))
        actor_obs = torch.cat((short_history, es_vel, compressed_long_history), dim=-1)
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def export(self, path):
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

def export_policy(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    train_cfg_dict = class_to_dict(train_cfg)
    policy_cfg = train_cfg_dict["policy"]
    num_critic_obs = env_cfg.env.num_privileged_obs
    if env_cfg.terrain.measure_heights:
        num_critic_obs = env_cfg.env.c_frame_stack * (env_cfg.env.single_num_privileged_obs + env_cfg.terrain.num_height)
    num_short_obs = env_cfg.env.short_frame_stack * env_cfg.env.num_single_obs

    actor_critic_class = eval(train_cfg_dict["runner"]["policy_class_name"])
    actor_critic: ActorCriticExplicit = actor_critic_class(
        num_short_obs, env_cfg.env.num_single_obs, num_critic_obs, env_cfg.env.num_actions, **policy_cfg
    )

    # load policy
    log_root_encoder = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported_data')
    model_path = get_load_path(log_root_encoder, load_run=args.load_run, checkpoint=args.checkpoint)
    print("Load model from:", model_path)
    loaded_dict = torch.load(model_path)
    actor_critic.load_state_dict(loaded_dict["model_state_dict"])

    exported_policy = Export_explicit(actor_critic.actor,
                                      actor_critic.long_history,
                                      actor_critic.state_estimator,
                                      num_short_obs,
                                      policy_cfg["in_channels"],
                                      env_cfg.env.num_single_obs)
    current_date_time = datetime.now().strftime("%Y_%m_%d_%H_%M-%S")

    root_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                             train_cfg.runner.experiment_name, 'exported_policies',
                             current_date_time)

    os.makedirs(root_path, exist_ok=True)
    dir_name = 'policy_explicit.pt'
    path = os.path.join(root_path, dir_name)
    exported_policy.export(path)
    print("Export Policy to:", path)

if __name__ == "__main__":
    EXPORT_POLICY = True
    args = get_args()
    if args.load_run == None:
        args.load_run = -1
    if args.checkpoint == None:
        args.checkpoint = -1
    export_policy(args)