import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, defaultdict
import time
import matplotlib.pyplot as plt
from pathlib import Path

ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 500
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5000
MAX_EPISODES = 1024
MAX_STEPS = 500
MODEL_PATH = "./model/dqn_cartpole.pth"
BEST_MODEL_PATH = "./model/best_dqn_cartpole.pth"
RES_PATH: str = "./RESULT"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)

def epsilon_by_frame(frame_idx):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * frame_idx / EPS_DECAY)

def train():
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(obs_dim, n_actions).to(device)
    target_net = DQN(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    frame_idx = 0

    rewards_per_episode = []  # List to track rewards per episode
    epsilons_per_episode = []  # List to track epsilon values per episode

    best_total_reward = -float('inf')
    best_model_state_dict = None

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        total_reward = 0

        epsilon: float = 0.0
        for step in range(MAX_STEPS):
            epsilon = epsilon_by_frame(frame_idx)
            frame_idx += 1

            state_v = torch.FloatTensor(state).unsqueeze(0).to(device)

            if random.random() > epsilon:
                with torch.no_grad():
                    q_values = policy_net(state_v)
                    action = q_values.argmax().item()
            else:
                action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                states_v = torch.FloatTensor(states).to(device)
                actions_v = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_v = torch.FloatTensor(rewards).to(device)
                next_states_v = torch.FloatTensor(next_states).to(device)
                dones_v = torch.BoolTensor(dones).to(device)

                q_values = policy_net(states_v).gather(1, actions_v).squeeze(1)
                with torch.no_grad():
                    max_next_q_values = target_net(next_states_v).max(1)[0]
                    target = rewards_v + GAMMA * max_next_q_values * (~dones_v)

                loss = nn.MSELoss()(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if frame_idx % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        rewards_per_episode.append(total_reward)  # Collect total reward per episode
        epsilons_per_episode.append(epsilon)  # Collect epsilon per episode

        # Save the best model based on total reward
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_model_state_dict = policy_net.state_dict()
            torch.save(best_model_state_dict, BEST_MODEL_PATH)
            print(f"New best model saved with total reward: {total_reward:.2f}")

        print(f"Episode {episode+1}, Total reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    # Save the final model (even if it's not the best)
    torch.save(policy_net.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    env.close()

    return rewards_per_episode, epsilons_per_episode

def plot_results(rewards_per_episode: list, epsilons_per_episode: list) -> None:
    # Plot total rewards per episode
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')

    # Plot epsilon decay
    plt.subplot(1, 2, 2)
    plt.plot(epsilons_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')

    plt.tight_layout()
    plt.savefig(Path(RES_PATH).joinpath("train_result.png"))
    plt.show()

def play():
    env = gym.make(ENV_NAME, render_mode="human")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(obs_dim, n_actions).to(device)
    policy_net.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))  # Load the best model
    policy_net.eval()

    state, _ = env.reset()
    env.render()  # 强制显示首帧

    done = False
    total_reward = 0

    while not done:
        state_v = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_v)
            action = q_values.argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)
        env.render()  # 每步渲染
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.02)

    print(f"Play finished. Total reward: {total_reward}")
    time.sleep(5)  # 窗口保持5秒
    env.close()

if __name__ == "__main__":
    # rewards_per_episode, epsilons_per_episode = train()  # 训练并返回结果
    # plot_results(rewards_per_episode, epsilons_per_episode)  # 绘制结果图
    play()
