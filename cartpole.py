from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from tinygrad.dtype import dtypes
import gymnasium as gym
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import imageio
import os

dtypes.default_float = dtypes.float32

class CustomLinear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor(np.random.uniform(-0.05, 0.05, (in_features, out_features)).astype(np.float32))
        self.bias = Tensor(np.zeros(out_features, dtype=np.float32))
    
    def __call__(self, x):
        return x.dot(self.weight) + self.bias

class DuelingQNetwork:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.shared1 = CustomLinear(state_dim, hidden_dim)
        self.shared2 = CustomLinear(hidden_dim, hidden_dim)
        
        self.value1 = CustomLinear(hidden_dim, hidden_dim // 2)
        self.value2 = CustomLinear(hidden_dim // 2, 1)
        
        self.advantage1 = CustomLinear(hidden_dim, hidden_dim // 2)
        self.advantage2 = CustomLinear(hidden_dim // 2, action_dim)
    
    def __call__(self, x):
        x = self.shared1(x).relu()
        x = self.shared2(x).relu()
        
        value = self.value1(x).relu()
        value = self.value2(value)
        
        advantage = self.advantage1(x).relu()
        advantage = self.advantage2(advantage)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        advantage_mean = advantage.mean(axis=1, keepdim=True)
        q_values = value + advantage - advantage_mean
        
        return q_values
    
    def load_state_dict(self, source_net):
        self.shared1.weight.assign(source_net.shared1.weight.detach())
        self.shared1.bias.assign(source_net.shared1.bias.detach())
        self.shared2.weight.assign(source_net.shared2.weight.detach())
        self.shared2.bias.assign(source_net.shared2.bias.detach())
        
        self.value1.weight.assign(source_net.value1.weight.detach())
        self.value1.bias.assign(source_net.value1.bias.detach())
        self.value2.weight.assign(source_net.value2.weight.detach())
        self.value2.bias.assign(source_net.value2.bias.detach())
        
        self.advantage1.weight.assign(source_net.advantage1.weight.detach())
        self.advantage1.bias.assign(source_net.advantage1.bias.detach())
        self.advantage2.weight.assign(source_net.advantage2.weight.detach())
        self.advantage2.bias.assign(source_net.advantage2.bias.detach())

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, target_update=50):
        self.q_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net)
        
        self.optimizer = optim.Adam([
            self.q_net.shared1.weight, self.q_net.shared1.bias,
            self.q_net.shared2.weight, self.q_net.shared2.bias,
            self.q_net.value1.weight, self.q_net.value1.bias,
            self.q_net.value2.weight, self.q_net.value2.bias,
            self.q_net.advantage1.weight, self.q_net.advantage1.bias,
            self.q_net.advantage2.weight, self.q_net.advantage2.bias
        ], lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_dim = action_dim
        self.target_update = target_update
        self.step_count = 0
        self.rewards = deque(maxlen=100)
        self.episode_count = 0
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = Tensor(state.astype(np.float32)).unsqueeze(0)
        q_values = self.q_net(state)
        return q_values.argmax().item()
    
    def shaped_reward(self, state, action, reward, next_state, done):
        if done and reward > 0:
            return reward
        elif done:
            return -10
        else:
            x, x_dot, theta, theta_dot = state
            
            angle_reward = 1.0 - abs(theta) / 0.2095
            position_reward = 1.0 - abs(x) / 2.4
            velocity_penalty = -0.1 * (abs(x_dot) + abs(theta_dot)) / 10.0
            survival_reward = 1.0
            
            total_reward = survival_reward + 0.1 * angle_reward + 0.1 * position_reward + velocity_penalty
            return max(total_reward, 0.1)
    
    def update(self, batch):
        self.step_count += 1
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = Tensor(np.array(states, dtype=np.float32))
        actions = Tensor(np.array(actions, dtype=np.int32))
        rewards = Tensor(np.array(rewards, dtype=np.float32))
        next_states = Tensor(np.array(next_states, dtype=np.float32))
        dones = Tensor(np.array(dones, dtype=np.float32))
        
        q_values = self.q_net(states)
        q_values_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN: use main network for action selection, target network for evaluation
        original_training = Tensor.training
        Tensor.training = False
        
        next_q_values_main = self.q_net(next_states)
        next_actions = next_q_values_main.argmax(axis=1)
        
        next_q_values_target = self.target_net(next_states)
        next_q_values_selected = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze()
        
        targets = rewards + self.gamma * next_q_values_selected * (1 - dones)
        
        Tensor.training = original_training
        
        loss = (q_values_selected - targets.detach()).pow(2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        for param in [self.q_net.shared1.weight, self.q_net.shared1.bias,
                     self.q_net.shared2.weight, self.q_net.shared2.bias,
                     self.q_net.value1.weight, self.q_net.value1.bias,
                     self.q_net.value2.weight, self.q_net.value2.bias,
                     self.q_net.advantage1.weight, self.q_net.advantage1.bias,
                     self.q_net.advantage2.weight, self.q_net.advantage2.bias]:
            if param.grad is not None:
                param.grad = param.grad.clip(-1.0, 1.0)
        
        self.optimizer.step()
        
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net)
        
        # Adaptive epsilon decay
        if self.episode_count < 50:
            self.epsilon = max(self.min_epsilon, self.epsilon * 0.998)
        else:
            self.epsilon = max(self.min_epsilon, self.epsilon * 0.99)
    
    def save_gif(self, env, filename="cartpole.gif", max_steps=500):
        frames = []
        state, _ = env.reset()
        done = False
        steps = 0
        original_training = Tensor.training
        Tensor.training = False
        while not done and steps < max_steps:
            frames.append(env.render())
            action = self.act(state)
            state, _, done, truncated, _ = env.step(action)
            done = done or truncated
            steps += 1
        Tensor.training = original_training
        env.close()
        imageio.mimsave(filename, frames, fps=30)

def train_dqn():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    lr = 0.001
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    target_update = 50
    batch_size = 64
    capacity = 20000
    episodes = 5000
    
    agent = DQNAgent(state_dim, action_dim, lr=lr, gamma=gamma, epsilon=epsilon, 
                    epsilon_decay=epsilon_decay, min_epsilon=min_epsilon, target_update=target_update)
    buffer = ReplayBuffer(capacity)
    rewards = []
    
    original_training_state = Tensor.training
    Tensor.training = True
    
    try:
        for ep in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            total_shaped_reward = 0
            done = False
            agent.episode_count = ep
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                
                shaped_reward = agent.shaped_reward(state, action, reward, next_state, done)
                total_shaped_reward += shaped_reward
                
                buffer.push(state, action, shaped_reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    agent.update(batch)
            
            agent.rewards.append(total_reward)
            rewards.append(total_reward)
            avg_reward = np.mean(list(agent.rewards)[-100:]) if len(agent.rewards) >= 100 else np.mean(agent.rewards)
            
            print(f"Episode {ep+1}, Reward: {total_reward}, Shaped: {total_shaped_reward:.1f}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            
            if len(agent.rewards) >= 100 and avg_reward >= 195:
                print("CartPole solved! Average reward >= 195 over 100 consecutive trials.")
                break
                
    finally:
        Tensor.training = original_training_state
    
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Dueling Double DQN with Reward Shaping")
    plt.savefig("dueling_dqn_reward_plot.png")
    plt.close()
    
    render_env = gym.make("CartPole-v1", render_mode="rgb_array")
    agent.save_gif(render_env, "dueling_dqn_cartpole.gif")
    
    env.close()
    return rewards

if __name__ == "__main__":
    train_dqn()