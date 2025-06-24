# RL-Dueling-Double-DQN
Using Dueling Double Deep Q Networks with reward shaping to solve CartPole-v1. Implemented using tinygrad.

# Dueling Double Deep Q Networks and Reward Shaping:

Combines Double DQN (DeepMind 2015) with Dueling Networks (DeepMind 2016) for improved learning efficiency.
The dueling architecture separates value estimation V(s) from advantage estimation A(s,a), allowing the network to learn state values independently of action-specific advantages.
Double DQN decouples action selection from target Q-value estimation to reduce overestimation bias common in vanilla DQN.
Experience replay reduces correlation between consecutive training samples and improves data efficiency.
Custom reward shaping provides dense feedback signals (pole angle, cart position, velocity penalties) instead of sparse survival rewards.

# Implementation Details:

CartPole-v1 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
Built entirely on tinygrad - RL implementation without PyTorch/TensorFlow.
Custom linear layers and network architecture implemented from scratch using tinygrad primitives.
Target network uses hard updates every 50 steps rather than soft updates for stability.
Adaptive epsilon decay: slower initial exploration (0.998 decay) for first 50 episodes, then faster exploitation (0.99 decay).
Gradient clipping (±1.0) prevents exploding gradients during early training phases.
Batch size of 64 with replay buffer capacity of 20,000 experiences.

# Results:

CartPole-v1 consistently solved in 100-120 episodes across multiple runs.
Reward shaping accelerates learning compared to vanilla sparse rewards.
The dueling architecture shows faster convergence than standard DQN implementations.

![image](https://github.com/user-attachments/assets/c53ad6f4-be71-4a32-862e-bba1ce0a4e39)
![image](https://github.com/user-attachments/assets/0f328fac-7d91-4419-8c88-0250f77bb19c)
