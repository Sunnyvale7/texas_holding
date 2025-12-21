import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict, Any
import os

# 导入我们的引擎组件
from holdem_server import (
    GameState, HoldemEngine, Seat, Action, ActType, Street, 
    Card, FeatureExtractor, MediumAgent, SimpleAgent
)

# ----------------------------
# 1. 神经网络模型定义
# ----------------------------
class PokerPolicyNet(nn.Module):
    def __init__(self, input_dim=27, output_dim=6): # 升级为 27 维 (引入后手胜率)
        super(PokerPolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256) # 拓宽网络以处理更复杂特征
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, output_dim)
        self.value_head = nn.Linear(128, 1) # 用于 Actor-Critic

    def forward(self, x):
        # 确保输入至少是 2D (batch_size, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # 动作概率分布
        probs = F.softmax(self.action_head(x), dim=-1)
        # 状态价值
        state_value = self.value_head(x)
        return probs, state_value

# ----------------------------
# 2. 动作映射
# ----------------------------
# 0: Fold, 1: Check/Call, 2: Raise 1/3, 3: Raise 1/2, 4: Raise 1x, 5: All-in
ACTION_MAP = {
    0: ActType.FOLD,
    1: ActType.CALL, # 引擎会自动转为 CHECK 如果 to_call 为 0
    2: "RAISE_0.33",
    3: "RAISE_0.5",
    4: "RAISE_1.0",
    5: ActType.ALL_IN
}

def map_idx_to_action(idx: int, legal_actions: List[Dict], state: Dict) -> Action:
    # 转换为引擎动作
    act_type_val = ACTION_MAP.get(idx)
    pot = state["public"]["pot_total"]
    
    # 查找合法动作
    acts = {a["type"]: a for a in legal_actions}
    
    if idx == 0: return Action(ActType.FOLD)
    
    if idx == 1:
        if ActType.CALL.value in acts: return Action(ActType.CALL, to_amount=acts[ActType.CALL.value]["to_amount"])
        if ActType.CHECK.value in acts: return Action(ActType.CHECK)
        return Action(ActType.FOLD) # 兜底
        
    if idx == 5:
        if ActType.ALL_IN.value in acts: return Action(ActType.ALL_IN)
        return Action(ActType.FOLD)

    # 处理加注档位
    target_type = ActType.RAISE.value if ActType.RAISE.value in acts else ActType.BET.value
    if target_type in acts:
        info = acts[target_type]
        ratio = float(act_type_val.split("_")[1])
        to_amt = info["min_to"] + int(pot * ratio)
        # 先圆整到 10
        to_amt = round(to_amt / 10) * 10
        # 再进行边界裁剪，确保不会超过 stack
        to_amt = max(info["min_to"], min(info["max_to"], to_amt))
        return Action(ActType(target_type), to_amount=to_amt)
    
    # 如果没法加注，尝试 Call/Check
    if ActType.CALL.value in acts: return Action(ActType.CALL, to_amount=acts[ActType.CALL.value]["to_amount"])
    if ActType.CHECK.value in acts: return Action(ActType.CHECK)
    return Action(ActType.FOLD)

# ----------------------------
# 3. 训练环境封装
# ----------------------------
class TrainingEnv:
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
        
        # 混合对手：1/3 中级，2/3 简单。这样 RL 更有可能赢到钱，从而获得正向反馈
        self.opponents = {}
        for i in range(1, self.num_seats):
            p_id = f"p{i}"
            if random.random() < 0.4:
                self.opponents[p_id] = MediumAgent(f"Player_{i}")
            else:
                self.opponents[p_id] = SimpleAgent(f"Player_{i}")

    def play_hand(self, model, device):
        # 记录开始时的筹码，用于计算奖励
        initial_stack = self.state.seats[0].stack
        
        # 如果筹码太少，自动补满但给予惩罚（模拟破产）
        bankruptcy_penalty = 0
        if initial_stack < self.state.bb:
            self.state.seats[0].stack = 2000
            initial_stack = 2000
            bankruptcy_penalty = -20.0 # 降低破产惩罚，避免模型过度胆小
            
        # 同时也检查并补满其他对手的筹码，确保游戏能进行
        for s in self.state.seats:
            if s.stack < self.state.bb:
                s.stack = 2000

        self.engine.start_hand()
        trajectories = [] # 存储 (obs, action_idx, log_prob, reward)
        
        while self.state.street != Street.HAND_OVER:
            seat_i = self.state.to_act
            if seat_i is None: break
            
            p_id = self.state.seats[seat_i].player_id
            
            # 生成当前玩家的私有状态
            # 这里需要模拟 holdem_server 中的 make_private_state
            from holdem_server import make_private_state
            # 注意：make_private_state 需要一个 Room 对象，这里我们做一个 Mock
            class MockRoom:
                def __init__(self, state, engine):
                    self.state = state
                    self.engine = engine
                    self.room_id = "train"
                    self.player_seat = {s.player_id: i for i, s in enumerate(state.seats)}
                    self.ready = set(s.player_id for s in state.seats)
            
            mock_room = MockRoom(self.state, self.engine)
            private_state = make_private_state(mock_room, p_id)
            
            if p_id == self.rl_player_id:
                # RL Agent 行动
                obs = FeatureExtractor.extract(private_state)
                obs_t = torch.FloatTensor(obs).to(device)
                
                probs, _ = model(obs_t)
                dist = torch.distributions.Categorical(probs[0]) # 取第一个 batch
                action_idx = dist.sample()
                
                action = map_idx_to_action(action_idx.item(), private_state["legal_actions"], private_state)
                
                trajectories.append({
                    "obs": obs,
                    "action_idx": action_idx,
                    "log_prob": dist.log_prob(action_idx),
                    "seat_i": seat_i,
                    "stack_before": self.state.seats[seat_i].stack + self.state.seats[seat_i].contributed_total
                })
                
                self.engine.apply_action(seat_i, action)
            else:
                # 机器人行动
                agent = self.opponents[p_id]
                action = agent.act(private_state)
                self.engine.apply_action(seat_i, action)

        # 结算奖励
        # 奖励 = 这一局的筹码变化量
        final_stack = self.state.seats[0].stack
        reward = (final_stack - initial_stack) / 50.0 + bankruptcy_penalty
        
        for t in trajectories:
            t["reward"] = reward
            
        return trajectories

# ----------------------------
# 4. 主训练循环
# ----------------------------
def train():
    # 强制使用 CPU 进行训练，避免小模型的 GPU 通讯开销
    device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    model = PokerPolicyNet().to(device)
    model_path = "models/poker_rl_latest.pth"
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            # 检查维度是否匹配 (fc1.weight 的输入维度应该是 27)
            if state_dict['fc1.weight'].shape[1] == 27:
                model.load_state_dict(state_dict)
                print(f"Loaded existing model from {model_path} to continue training...", flush=True)
            else:
                print(f"Architecture mismatch (Old: {state_dict['fc1.weight'].shape[1]} dims, New: 27 dims). Starting fresh.", flush=True)
        except Exception as e:
            print(f"Could not load model: {e}, starting from scratch.", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=5e-5) # 降低学习率，更稳定
    env = TrainingEnv(num_seats=6)

    running_reward = 0
    batch_loss = []
    batch_size = 32 # 攒够32局再更新权重
    
    print("Starting training... (Target: 100,000+ episodes)", flush=True)
    
    if not os.path.exists("models"):
        os.makedirs("models")

    from tqdm import tqdm
    pbar = tqdm(range(1, 200000), desc="Standard Training")
    for episode in pbar:
        # 每 50 手牌或者破产后强制重置一次环境，防止状态过于离散
        if episode % 50 == 0:
            env.reset()
            
        trajectories = env.play_hand(model, device)
        
        if not trajectories: continue

        reward = trajectories[0]["reward"]
        running_reward = 0.999 * running_reward + 0.001 * reward # 更平滑的趋势
        
        for t in trajectories:
            # 1. 策略梯度损失
            # 计算优势 (Advantage): 实际收益 - 期望收益
            advantage = t["reward"] - running_reward
            
            # 2. 引入熵损失 (Entropy Loss): 鼓励探索，防止坍缩为“只会弃牌”
            # 提高权重到 0.05，强制模型探索更多动作
            probs, _ = model(torch.FloatTensor(t["obs"]).to(device))
            dist = torch.distributions.Categorical(probs[0])
            entropy = dist.entropy()
            
            loss = -t["log_prob"] * advantage - 0.05 * entropy
            batch_loss.append(loss)
        
        # 达到批次大小后更新
        if episode % batch_size == 0:
            optimizer.zero_grad()
            total_loss = torch.stack(batch_loss).mean()
            total_loss.backward()
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_loss = []

        if episode % 200 == 0:
            pbar.set_postfix({
                "Trend": f"{running_reward:.4f}",
                "Last": f"{reward:.2f}"
            })
            torch.save(model.state_dict(), f"models/poker_rl_latest.pth")

if __name__ == "__main__":
    train()

