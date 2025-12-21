import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict, Any
import os

from holdem_server import (
    GameState, HoldemEngine, Seat, Action, ActType, Street, 
    Card, FeatureExtractor, MediumAgent, SimpleAgent
)
# 引用基础训练脚本中的网络结构和动作映射
from train_rl import PokerPolicyNet, map_idx_to_action, TrainingEnv, ACTION_MAP

class HardTrainingEnv(TrainingEnv):
    def reset(self):
        seats = []
        for i in range(self.num_seats):
            seats.append(Seat(player_id=f"p{i}", name=f"Player_{i}", stack=2000, hole=[]))
        self.state = GameState(seats=seats, sb=10, bb=20, button=random.randint(0, self.num_seats-1))
        self.engine = HoldemEngine(self.state)
        self.rl_player_id = "p0"
        
        # 困难模式：100% 对手都是中级机器人
        self.opponents = {f"p{i}": MediumAgent(f"Player_{i}") for i in range(1, self.num_seats)}

def train_hard():
    # 强制使用 CPU 进行训练，避免小模型的 GPU 通讯开销
    device = torch.device("cpu")
    print(f"Using device: {device} | HARD MODE: VS 5 MediumAgents", flush=True)

    model = PokerPolicyNet().to(device)
    model_path = "models/poker_rl_latest.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Loaded existing model to continue hard training...")

    optimizer = optim.Adam(model.parameters(), lr=3e-5) # 更低的学习率，防止波动
    env = HardTrainingEnv(num_seats=6)

    running_reward = 0
    batch_loss = []
    batch_size = 32
    
    from tqdm import tqdm
    pbar = tqdm(range(1, 100000), desc="Hard Mode Training")
    for episode in pbar:
        if episode % 50 == 0:
            env.reset()
            
        trajectories = env.play_hand(model, device)
        if not trajectories: continue

        reward = trajectories[0]["reward"]
        running_reward = 0.999 * running_reward + 0.001 * reward
        
        for t in trajectories:
            advantage = t["reward"] - running_reward
            probs, _ = model(torch.FloatTensor(t["obs"]).to(device))
            dist = torch.distributions.Categorical(probs[0])
            entropy = dist.entropy()
            
            loss = -t["log_prob"] * advantage - 0.02 * entropy # 稍微调高熵权重，鼓励在高手对决中探索
            batch_loss.append(loss)
        
        if episode % batch_size == 0:
            optimizer.zero_grad()
            torch.stack(batch_loss).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_loss = []

        if episode % 200 == 0:
            pbar.set_postfix({
                "Trend": f"{running_reward:.4f}",
                "Last": f"{reward:.2f}"
            })
            torch.save(model.state_dict(), f"models/poker_rl_latest.pth")

if __name__ == "__main__":
    train_hard()

