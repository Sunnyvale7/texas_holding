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
    Card, FeatureExtractor, BaseAgent, make_private_state
)
from train_rl import PokerPolicyNet, map_idx_to_action, ACTION_MAP

class SelfPlayAgent(BaseAgent):
    """使用当前模型进行决策的对手"""
    def __init__(self, name: str, model, device):
        super().__init__(name)
        self.model = model
        self.device = device

    def act(self, state: Dict[str, Any]) -> Action:
        obs = FeatureExtractor.extract(state)
        obs_t = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            probs, _ = self.model(obs_t)
            # 自博弈时，我们可以加入一点随机性
            dist = torch.distributions.Categorical(probs[0])
            action_idx = dist.sample().item()
        return map_idx_to_action(action_idx, state["legal_actions"], state)

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
        self.engine.start_hand()
        trajectories = []
        
        while self.state.street != Street.HAND_OVER:
            seat_i = self.state.to_act
            if seat_i is None: break
            p_id = self.state.seats[seat_i].player_id
            
            # Mock Room for make_private_state
            class MockRoom:
                def __init__(self, state, engine):
                    self.state, self.engine = state, engine
                    self.room_id = "self-play"
                    self.player_seat = {s.player_id: i for i, s in enumerate(state.seats)}
                    self.ready = set(s.player_id for s in state.seats)
            
            private_state = make_private_state(MockRoom(self.state, self.engine), p_id)
            
            if seat_i == self.train_idx:
                obs = FeatureExtractor.extract(private_state)
                obs_t = torch.FloatTensor(obs).to(device)
                probs, _ = model(obs_t)
                dist = torch.distributions.Categorical(probs[0])
                action_idx = dist.sample()
                action = map_idx_to_action(action_idx.item(), private_state["legal_actions"], private_state)
                
                trajectories.append({
                    "obs": obs, "action_idx": action_idx, "log_prob": dist.log_prob(action_idx),
                    "seat_i": seat_i
                })
                self.engine.apply_action(seat_i, action)
            else:
                action = self.opponents[p_id].act(private_state)
                self.engine.apply_action(seat_i, action)

        # 奖励计算：这一局的盈亏
        final_stack = self.state.seats[self.train_idx].stack
        reward = (final_stack - 2000) / 20.0
        for t in trajectories: t["reward"] = reward
        return trajectories

def train_self_play():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} | MODE: SELF-PLAY")

    model = PokerPolicyNet().to(device)
    model_path = "models/poker_rl_latest.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    env = SelfPlayEnv(num_seats=6)
    running_reward = 0
    batch_loss, batch_size = [], 64 # 自博弈需要更大的 batch 来保证稳定

    for episode in range(1, 500000):
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
            # 自博弈的 Trend 理论上会趋向于 0（因为大家水平一样），所以观察的是策略的稳定性
            print(f"Self-Play Episode {episode}, Trend (Relative): {running_reward:.4f}")
            torch.save(model.state_dict(), f"models/poker_rl_latest.pth")

if __name__ == "__main__":
    train_self_play()

