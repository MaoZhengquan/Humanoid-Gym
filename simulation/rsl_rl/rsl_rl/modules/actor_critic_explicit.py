import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

class ActorCriticExplicit(nn.Module):
    def __init__(self, num_short_obs,
                       num_proprio_obs,
                       num_critic_obs,
                       num_actions,
                       actor_hidden_dims=[256, 256, 256],
                       critic_hidden_dims=[256,256,256],
                       state_estimator_hidden_dims=[256, 128, 64],
                       in_channels = 66,
                       kernel_size=[6, 4],
                       filter_size=[32, 16],
                       stride_size=[3, 2],
                       lh_output_dim=64,
                       init_noise_std=1.0,
                       fix_action_std=True,
                       action_std=None,
                       activation='elu',
                       **kwargs):

        if kwargs:
            print("ActorCriticExplicit.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))

        if fix_action_std:
            assert action_std is not None, "action_std must be provided if you need to fix action std!"

        super().__init__()

        self.fix_action_std = fix_action_std
        print("self.fix_action_std",self.fix_action_std)

        # 定义actor和critic网络
        # num_short_obs = num_single_obs * short_frame_stack 5 history
        # lh_output_dim是cnn的输出
        # 状态估计的输出是3维
        mlp_input_dim_a = num_short_obs + lh_output_dim + 3
        mlp_input_dim_c = num_critic_obs
        activation = get_activation(activation)
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a,actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
            # num actions policy output
                actor_layers.append(nn.Linear(actor_hidden_dims[l],num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l],actor_hidden_dims[l+1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value Function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c,critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l],1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l],critic_hidden_dims[l+1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        if self.fix_action_std:
            action_std_tensor = torch.tensor(action_std)
            self.std = nn.Parameter(action_std_tensor,requires_grad=False)
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        print("std",self.std)
        self.distribution = None
        Normal.set_default_validate_args = False

        # define long_history CNN
        long_history_layers = []
        self.in_channels = in_channels
        cnn_output_dim = num_proprio_obs
        for out_channels, kernel_size, stride_size in zip(filter_size, kernel_size, stride_size):
            long_history_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride_size))
            long_history_layers.append(nn.ReLU())
            cnn_output_dim = (cnn_output_dim - kernel_size + stride_size) // stride_size
            in_channels = out_channels
        cnn_output_dim *= out_channels
        long_history_layers.append(nn.Flatten())
        long_history_layers.append(nn.Linear(cnn_output_dim,128))
        long_history_layers.append(nn.ELU())
        long_history_layers.append(nn.Linear(128, lh_output_dim))
        self.long_history = nn.Sequential(*long_history_layers)
        print(f"long_history CNN: {self.long_history}")

        # define state_estimator MLP
        self.num_short_obs = num_short_obs
        state_estimator_input_dim = num_short_obs
        state_estimator_output_dim = 3 # 估计的线速度
        state_estimator_layers = []
        state_estimator_layers.append(nn.Linear(state_estimator_input_dim,state_estimator_hidden_dims[0]))
        state_estimator_layers.append(activation)

        for l in range(len(state_estimator_hidden_dims)):
            if l == len(state_estimator_hidden_dims) - 1:
                state_estimator_layers.append(nn.Linear(state_estimator_hidden_dims[l],state_estimator_output_dim))
            else:
                state_estimator_layers.append(nn.Linear(state_estimator_hidden_dims[l],state_estimator_hidden_dims[l+1]))
                state_estimator_layers.append(activation)
        self.state_estimator = nn.Sequential(*state_estimator_layers)
        print(f"state_estimator MLP: {self.state_estimator}")

        self.num_proprio_obs = num_proprio_obs

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
        enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, **kwargs):
        short_history = observations[...,-self.num_short_obs:]
        es_vel = self.state_estimator(short_history)
        compressed_long_history = self.long_history(observations.view(-1, self.in_channels, self.num_proprio_obs))
        actor_obs = torch.cat((short_history, es_vel, compressed_long_history),dim=-1)
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        short_history = observations[...,-self.num_short_obs:]
        es_vel = self.state_estimator(short_history)
        compressed_long_history = self.long_history(observations.view(-1, self.in_channels, self.num_proprio_obs))
        actor_obs = torch.cat((short_history, es_vel, compressed_long_history),dim=-1)
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None