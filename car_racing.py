import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Callable, Any, Union
from collections import defaultdict
from tqdm import tqdm


"""
    Define the Critic and value network, with CNN as the backbone network
    to extract the feature of the input image. In the gym environment, the
    program will capture one figure which resolution is 3*96*96 as the
    feature.
"""
class ActorCriticCNN(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        self.fc_input_dim = self._get_conv_out((3, 96, 96))
        self.fc = nn.Sequential(nn.Linear(self.fc_input_dim, 512), nn.ReLU())
        self.actor_mean = nn.Linear(512, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(512, 1)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv(o)
        return int(np.prod(o.size()))

    def forward(self, x) -> tuple[torch.Tensor, dict[str, Any]]:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # action_head
        mean: torch.Tensor = self.actor_mean(x)
        # 先取log再exp， 保证为正数
        std: torch.Tensor = self.actor_log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        action: torch.Tensor = distribution.sample()
        # 该动作的概率
        log_prob: torch.Tensor = distribution.log_prob(action).sum(dim=-1)

        # value head
        value: torch.Tensor = self.critic(x)

        return value, dict(action=action, distri=distribution, log_prob=log_prob)


# np.ndarray -> torch.Tensor
def transform(
        array: Union[np.ndarray, list, torch.Tensor]
) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32)

class Play(object):
    def __init__(self, policy: torch.nn.Module, device: Union[str, torch.device] = "cpu"):
        self.policy = policy
        self.env = gym.make("CarRacing-v3")
        self.device = torch.device(device)

    def play(self, max_timesteps: int) -> dict:
        memory: defaultdict[str, list] = defaultdict(list)
        state, _ = self.env.reset()
        state = transform(array=state).permute(2, 0, 1).unsqueeze(dim=0)

        state = state.to(device=self.device)
        pbar = tqdm(desc="Gather_Data", total=max_timesteps, ncols=100)
        for episode in range(max_timesteps):
            value, action_dict = self.policy(state)

            action = action_dict["action"]
            action = torch.clamp(action, -1.0, 1.0)
            action_numpy: np.ndarray = action.cpu().detach().squeeze().numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(action=action_numpy)
            done = terminated or truncated
            memory['states'].append(state.squeeze(0))
            memory['actions'].append(action.squeeze(0))
            memory['logprobs'].append(action_dict["log_prob"])
            memory['rewards'].append(reward)
            memory['dones'].append(done)

            state = transform(array=next_state).permute(2, 0, 1).unsqueeze(dim=0)

            pbar.update(1)
            if done: break

        return memory


"""
    GAE估计公式：
    A_t^{\text{GAE}} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \
    (r_{t+k} + \gamma V(s_{t+k+1}) - V(s_{t+k}))
"""
class GAE(object):
    def __init__(self, gamma: float = 0.99, lambda_: float = 0.95):
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(
            self, rewards: torch.Tensor, values: torch.Tensor,
            dones: torch.Tensor) -> torch.Tensor:
        advantages: torch.Tensor = torch.zeros_like(rewards)

        next_value: torch.Tensor = torch.as_tensor(data=0)
        last_advantage: torch.Tensor = torch.as_tensor(data=0)

        for t in reversed(range(rewards.shape[0])):
            # 计算TD error, 对于终止状态，下一个节点的奖励至是0
            delta: torch.Tensor = rewards[t] + self.gamma*next_value*(1-dones[t]) - values[t]
            # 更新GAE value
            advantages[t] = delta + self.gamma*self.lambda_*last_advantage

            # 更新next_value 和 last_advantage
            next_value = values[t]
            last_advantage = advantages[t]

        return advantages

class PPO(object):
    def __init__(
            self, action_dim: int = 3, device: str = "cpu",
            lr: float = 1e-3, gamma: float = 0.999
    ):
        self.device = torch.device(device)
        self.policy = ActorCriticCNN(action_dim).to(device)
        self.policy_old = ActorCriticCNN(action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
        self.gamma = gamma
        self.eps_clip: float = 0.2

        self.advantage_estimator: Callable = GAE(gamma=0.99, lambda_=0.95)

    def optimize(self, memory: dict, epoches: int = 4) -> None:
        # parse data from the dict
        states = torch.stack(memory['states']).to(self.device)
        actions = torch.stack(memory['actions']).to(self.device)
        old_logprobs = torch.stack(memory['logprobs']).to(self.device)
        rewards = memory['rewards']
        dones = memory['dones']

        # 计算折扣回报
        discounted_reward = []
        discounted: float = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            discounted = reward + self.gamma*discounted
            discounted_reward.insert(0, discounted)

        discounted_reward = torch.tensor(
            data=discounted_reward, dtype=torch.float32, device=self.device
        )

        for _ in range(epoches):
            values, action_dicts = self.policy(states)
            new_distri: torch.distributions.Normal = action_dicts["distri"]
            log_probs_new = new_distri.log_prob(value=actions).sum(dim=-1)

            # advantages = discounted_reward-values.detach().squeeze()
            # 使用GAE进行估计， 这样一来方差能够小一些
            advantages: torch.Tensor = self.advantage_estimator(
                rewards=transform(rewards), values=values.squeeze(),
                dones=transform(dones)
            )
            ratio = torch.exp(log_probs_new.squeeze()-old_logprobs.detach().squeeze())

            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1-self.eps_clip, 1+self.eps_clip)*advantages

            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * self.MseLoss(discounted_reward, values.squeeze())
            print(f"[K_EPOCH: {_+1}, LOSS: {loss.item():.4f}]")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

def train(
        ppo_config: dict, max_episodes: int,
        max_timesteps: int, epoches: int = 4) -> dict:
    ppo = PPO(**ppo_config)
    for episode in range(max_episodes):
        player = Play(policy=ppo.policy_old, device=ppo.device)
        memory: dict = player.play(max_timesteps=max_timesteps)
        ppo.optimize(memory=memory, epoches=epoches)

        reward: float = sum(memory["rewards"])
        print(f"Episode {episode} Reward: {reward:.2f}")

    return ppo.policy.state_dict()


if __name__ == "__main__":
    ppo_config: dict = dict(
        action_dim=3, device="cpu",
        lr=3e-4, gamma=0.99
    )
    train(
        ppo_config=ppo_config, max_episodes=128,
        max_timesteps=1024, epoches=4
    )