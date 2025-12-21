import torch
import torch.nn as nn
import numpy as np
import os
import random
from typing import List, Dict, Any

from holdem_server import (
    GameState, HoldemEngine, Seat, Action, ActType, Street, 
    Card, FeatureExtractor, SimpleAgent
)
from train_rl import PokerPolicyNet, map_idx_to_action

class EvaluationEnv:
    def __init__(self, num_seats=6):
        self.num_seats = num_seats
        self.reset()

    def reset(self):
        seats = []
        for i in range(self.num_seats):
            seats.append(Seat(player_id=f"p{i}", name=f"Player_{i}", stack=2000, hole=[]))
        self.state = GameState(seats=seats, sb=10, bb=20, button=random.randint(0, self.num_seats-1))
        self.engine = HoldemEngine(self.state)
        self.rl_player_id = "p0"
        # 全部对手都是简单机器人
        self.opponents = {f"p{i}": SimpleAgent(f"Player_{i}") for i in range(1, self.num_seats)}

    def play_hand(self, model, device):
        self.engine.start_hand()
        
        while self.state.street != Street.HAND_OVER:
            seat_i = self.state.to_act
            if seat_i is None: break
            
            p_id = self.state.seats[seat_i].player_id
            
            from holdem_server import make_private_state
            class MockRoom:
                def __init__(self, state, engine):
                    self.state = state
                    self.engine = engine
                    self.room_id = "eval"
                    self.player_seat = {s.player_id: i for i, s in enumerate(state.seats)}
                    self.ready = set(s.player_id for s in state.seats)
            
            mock_room = MockRoom(self.state, self.engine)
            private_state = make_private_state(mock_room, p_id)
            
            if p_id == self.rl_player_id:
                obs = FeatureExtractor.extract(private_state)
                obs_t = torch.FloatTensor(obs).to(device)
                
                with torch.no_grad():
                    probs, _ = model(obs_t)
                    # 测试时选择概率最大的动作 (Deterministic)
                    action_idx = torch.argmax(probs[0]).item()
                
                action = map_idx_to_action(action_idx, private_state["legal_actions"], private_state)
                self.engine.apply_action(seat_i, action)
            else:
                action = self.opponents[p_id].act(private_state)
                self.engine.apply_action(seat_i, action)

        return self.state.seats[0].stack - 2000

def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PokerPolicyNet().to(device)
    model_path = "models/poker_rl_latest.pth"
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            if state_dict['fc1.weight'].shape[1] == 27:
                model.load_state_dict(state_dict)
                print(f"Successfully loaded model: {model_path}")
            else:
                print(f"Architecture mismatch (Model: {state_dict['fc1.weight'].shape[1]} dims, Expected: 27). Using random weights.")
        except Exception as e:
            print(f"Error loading model: {e}. Using random weights.")
    else:
        print("No trained model found. Running with random weights.")

    model.eval()
    env = EvaluationEnv(num_seats=6)
    
    total_profit = 0
    wins = 0
    num_hands = 500
    
    print(f"Starting evaluation over {num_hands} hands vs 5 SimpleAgents...")
    
    for i in range(num_hands):
        env.reset()
        profit = env.play_hand(model, device)
        total_profit += profit
        if profit > 0:
            wins += 1
        
        if (i + 1) % 50 == 0:
            print(f"Hand {i+1}/{num_hands} | Total Profit: {total_profit} | Win Rate: {wins/(i+1)*100:.1f}%")

    print("\n" + "="*30)
    print("EVALUATION RESULT")
    print(f"Total Hands: {num_hands}")
    print(f"Total Profit: {total_profit} (Avg: {total_profit/num_hands:.2f} per hand)")
    print(f"Win Rate: {wins/num_hands*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    evaluate()


