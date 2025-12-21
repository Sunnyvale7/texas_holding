import sys
print("正在启动程序，请稍候 (加载 Torch 库可能需要 10-20 秒)...", flush=True)

import torch
print("Torch 库加载成功！", flush=True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict, Any
import os
from dataclasses import asdict
from tqdm import tqdm

from holdem_server import (
    GameState, HoldemEngine, Seat, Action, ActType, Street, 
    Card, FeatureExtractor, BaseAgent, make_private_state
)
from train_rl import PokerPolicyNet, map_idx_to_action, ACTION_MAP

class SelfPlayAgent(BaseAgent):
    """使用当前模型进行决策的对手"""
    def __init__(self, name: str, model, device):
        super().__init__(name)
        self.model = model
        self.device = device

    def act_fast(self, game_state, seat_idx) -> Action:
        obs = FeatureExtractor.extract_raw(game_state, seat_idx)
        obs_t = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            probs, _ = self.model(obs_t)
            dist = torch.distributions.Categorical(probs[0])
            action_idx = dist.sample().item()
        
        # 为了 map_idx_to_action 的兼容性，我们还是需要一个简化的 state 字典
        seat = game_state.seats[seat_idx]
        temp_state = {
            "public": {
                "pot_total": game_state.pot_total,
                "current_bet": game_state.current_bet,
            }
        }
        
        # 使用传入的 game_state 创建临时引擎获取合法动作
        engine = HoldemEngine(game_state)
        legal_actions = [asdict(a) for a in engine.legal_actions(seat_idx)]
        for a in legal_actions:
            a["type"] = a["type"].value if hasattr(a["type"], "value") else a["type"]
            
        return map_idx_to_action(action_idx, legal_actions, temp_state)

class SelfPlayEnv:
    def __init__(self, num_seats=6):
        self.num_seats = num_seats
        
    def reset(self, model, device):
        seats = []
        for i in range(self.num_seats):
            seats.append(Seat(player_id=f"p{i}", name=f"Player_{i}", stack=2000, hole=[]))
        self.state = GameState(seats=seats, sb=10, bb=20, button=random.randint(0, self.num_seats-1))
        self.engine = HoldemEngine(self.state)
        
        # 挑选一个随机位置作为训练对象，其他位置作为自博弈对手
        self.train_idx = random.randint(0, self.num_seats - 1)
        self.rl_player_id = f"p{self.train_idx}"
        
        # 所有对手都使用当前的模型
        self.opponents = {}
        for i in range(self.num_seats):
            if i != self.train_idx:
                self.opponents[f"p{i}"] = SelfPlayAgent(f"Player_{i}", model, device)

    def play_hand(self, model, device):
        # 记录开始时的筹码，用于计算奖励
        initial_stack = self.state.seats[self.train_idx].stack
        
        # 如果筹码太少，自动补满但给予惩罚（模拟破产）
        bankruptcy_penalty = 0
        if initial_stack < self.state.bb:
            self.state.seats[self.train_idx].stack = 2000
            initial_stack = 2000
            bankruptcy_penalty = -100.0 # 严重的破产惩罚
            
        # 同时也检查并补满其他对手的筹码，确保游戏能进行
        for s in self.state.seats:
            if s.stack < self.state.bb:
                s.stack = 2000

        self.engine.start_hand()
        trajectories = []
        
        while self.state.street != Street.HAND_OVER:
            seat_i = self.state.to_act
            if seat_i is None: break
            
            if seat_i == self.train_idx:
                obs = FeatureExtractor.extract_raw(self.state, seat_i)
                obs_t = torch.FloatTensor(obs).to(device)
                probs, _ = model(obs_t)
                dist = torch.distributions.Categorical(probs[0])
                action_idx = dist.sample()
                
                # 简化版状态用于映射动作
                temp_state = {"public": {"pot_total": self.state.pot_total, "current_bet": self.state.current_bet}}
                from dataclasses import asdict
                legal_actions = [asdict(a) for a in self.engine.legal_actions(seat_i)]
                for a in legal_actions: a["type"] = a["type"].value
                
                action = map_idx_to_action(action_idx.item(), legal_actions, temp_state)
                
                trajectories.append({
                    "obs": obs, "action_idx": action_idx, "log_prob": dist.log_prob(action_idx),
                    "seat_i": seat_i
                })
                self.engine.apply_action(seat_i, action)
            else:
                p_id = self.state.seats[seat_i].player_id
                action = self.opponents[p_id].act_fast(self.state, seat_i)
                self.engine.apply_action(seat_i, action)

        # 奖励计算：这一局的筹码变化量
        final_stack = self.state.seats[self.train_idx].stack
        # 奖励 = (结束筹码 - 开始筹码) / 50.0 + 破产惩罚
        reward = (final_stack - initial_stack) / 50.0 + bankruptcy_penalty
        for t in trajectories: t["reward"] = reward
        return trajectories

def train_self_play():
    # 强制使用 CPU 进行训练，避免小模型的 GPU 通讯开销
    device = torch.device("cpu")
    print(f"Using device: {device} | MODE: SELF-PLAY (FAST-PATH)", flush=True)

    print("正在初始化网络模型...", flush=True)
    model = PokerPolicyNet().to(device)
    model_path = "models/poker_rl_latest.pth"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"成功加载旧模型: {model_path}", flush=True)
        except Exception as e:
            print(f"加载模型失败: {e}, 将从零开始训练。", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    env = SelfPlayEnv(num_seats=6)
    running_reward = 0
    batch_loss, batch_size = [], 64
    
    if not os.path.exists("models"):
        os.makedirs("models")

    print("正在重置训练环境...", flush=True)
    env.reset(model, device)
    
    print("开始训练循环...", flush=True)
    pbar = tqdm(range(1, 500000), desc="Self-Play Training", mininterval=1.0)
    for episode in pbar:
        if episode % 50 == 0:
            env.reset(model, device)
            
        trajectories = env.play_hand(model, device)
        if not trajectories: continue

        reward = trajectories[0]["reward"]
        running_reward = 0.999 * running_reward + 0.001 * reward
        
        for t in trajectories:
            advantage = t["reward"] - 0 # 自博弈中，baseline 很难定义，通常设为 0
            loss = -t["log_prob"] * advantage
            batch_loss.append(loss)
        
        if episode % batch_size == 0:
            optimizer.zero_grad()
            torch.stack(batch_loss).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            batch_loss = []

        if episode % 500 == 0:
            # 更新进度条右侧的统计信息
            pbar.set_postfix({
                "Trend": f"{running_reward:.4f}",
                "Last": f"{reward:.2f}"
            })
            torch.save(model.state_dict(), f"models/poker_rl_latest.pth")

if __name__ == "__main__":
    train_self_play()

