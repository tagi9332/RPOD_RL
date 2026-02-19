import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
# Replace with the actual path to your CSV file
df = pd.read_csv("ppo_logs/progress.csv")

# 2. Clean the data (forward-fill missing values caused by TensorBoard's alternating log rows)
df_clean = df.fillna(method='ffill')

# 3. Create a 3-panel plot
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig.suptitle('PPO Agent Training Progress', fontsize=16)

# Plot 1: Mean Reward (The Learning Curve)
axs[0].plot(df_clean['time/total_timesteps'], df_clean['rollout/ep_rew_mean'], color='green', linewidth=2)
axs[0].set_title('Mean Episode Reward')
axs[0].set_ylabel('Reward')
axs[0].grid(True, alpha=0.3)

# Plot 2: Episode Length (Efficiency)
axs[1].plot(df_clean['time/total_timesteps'], df_clean['rollout/ep_len_mean'], color='blue', linewidth=2)
axs[1].set_title('Mean Episode Length')
axs[1].set_ylabel('Steps')
axs[1].grid(True, alpha=0.3)

# Plot 3: Frames Per Second (Performance)
axs[2].plot(df_clean['time/total_timesteps'], df_clean['time/fps'], color='red', linewidth=2)
axs[2].set_title('Simulation Speed (FPS)')
axs[2].set_xlabel('Total Timesteps')
axs[2].set_ylabel('FPS')
axs[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_progress.png")


# Second Figure: Explained Variance, Standard Deviation, and Value Loss
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig.suptitle('PPO Agent Training Metrics', fontsize=16)
# Plot 1: Explained Variance (Value Function Accuracy)
axs[0].plot(df_clean['time/total_timesteps'], df_clean['train/explained_variance'], color='purple', linewidth=2)
axs[0].set_title('Explained Variance (Value Function Accuracy)')
axs[0].set_ylabel('Explained Variance')
axs[0].grid(True, alpha=0.3)
# Plot 2: Standard Deviation of Rewards (Reward Variability)
axs[1].plot(df_clean['time/total_timesteps'], df_clean['train/std'], color='orange', linewidth=2)
axs[1].set_title('Standard Deviation of Episode Rewards')
axs[1].set_ylabel('Reward Std Dev')
axs[1].grid(True, alpha=0.3)
# Plot 3: Value Loss (Critic Learning Progress)
axs[2].plot(df_clean['time/total_timesteps'], df_clean['train/value_loss'], color='brown', linewidth=2)
axs[2].set_title('Value Loss (Critic Learning Progress)')
axs[2].set_xlabel('Total Timesteps')
axs[2].set_ylabel('Value Loss')
axs[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_metrics.png")