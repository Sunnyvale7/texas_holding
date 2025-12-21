# holdem_server.py
from __future__ import annotations

import asyncio
import json
import random
import os
from dataclasses import dataclass, asdict
from enum import Enum
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
async def get_index():
    with open("test_client.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ----------------------------
# Cards / Deck
# ----------------------------
RANKS = list(range(2, 15))  # 2..14 (A=14)
SUITS = ["s", "h", "d", "c"]  # spade/heart/diamond/club


@dataclass(frozen=True)
class Card:
    r: int
    s: str

    def __str__(self) -> str:
        face = {11: "J", 12: "Q", 13: "K", 14: "A"}.get(self.r, str(self.r))
        return f"{face}{self.s}"

    @staticmethod
    def from_str(x: str) -> "Card":
        # e.g. "As", "Th", "9d"
        face = x[:-1]
        s = x[-1]
        r = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10}.get(face, None)
        if r is None:
            r = int(face)
        return Card(r=r, s=s)


class Deck:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.cards = [Card(r, s) for r in RANKS for s in SUITS]
        self.rng.shuffle(self.cards)

    def deal(self, n: int) -> List[Card]:
        if len(self.cards) < n:
            raise RuntimeError("Deck out of cards")
        out = self.cards[:n]
        self.cards = self.cards[n:]
        return out


# ----------------------------
# Hand evaluator (7 cards -> best 5-card)
# Rank tuple: (category, tiebreakers...)
# Higher is better.
# category: 8=SF,7=4K,6=FH,5=F,4=S,3=3K,2=2P,1=1P,0=HC
# ----------------------------
def _is_straight(ranks: List[int]) -> Optional[int]:
    # ranks descending unique
    rs = sorted(set(ranks), reverse=True)
    # wheel straight A-5
    if set([14, 5, 4, 3, 2]).issubset(set(ranks)):
        return 5
    for i in range(len(rs) - 4):
        window = rs[i : i + 5]
        if window[0] - window[4] == 4 and len(window) == 5:
            return window[0]
    return None


HAND_CATEGORIES = {
    0: "高牌",
    1: "一对",
    2: "两对",
    3: "三条",
    4: "顺子",
    5: "同花",
    6: "葫芦",
    7: "四条",
    8: "同花顺"
}


def eval_5(cards: List[Card]) -> Tuple[int, Tuple[int, ...]]:
    ranks = sorted([c.r for c in cards], reverse=True)
    suits = [c.s for c in cards]
    flush = len(set(suits)) == 1

    counts: Dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # sort by (count desc, rank desc)
    groups = sorted(((cnt, r) for r, cnt in counts.items()), reverse=True)
    # e.g. [(3, 14), (1, 9), (1, 2)]
    straight_high = _is_straight(ranks)

    if flush and straight_high is not None:
        return (8, (straight_high,))  # straight flush
    if groups[0][0] == 4:
        four_rank = groups[0][1]
        kicker = max(r for r in ranks if r != four_rank)
        return (7, (four_rank, kicker))
    if groups[0][0] == 3 and groups[1][0] == 2:
        trips = groups[0][1]
        pair = groups[1][1]
        return (6, (trips, pair))
    if flush:
        return (5, tuple(ranks))
    if straight_high is not None:
        return (4, (straight_high,))
    if groups[0][0] == 3:
        trips = groups[0][1]
        kickers = sorted([r for r in ranks if r != trips], reverse=True)
        return (3, (trips, *kickers))
    if groups[0][0] == 2 and groups[1][0] == 2:
        p1, p2 = sorted([groups[0][1], groups[1][1]], reverse=True)
        kicker = max(r for r in ranks if r != p1 and r != p2)
        return (2, (p1, p2, kicker))
    if groups[0][0] == 2:
        pair = groups[0][1]
        kickers = sorted([r for r in ranks if r != pair], reverse=True)
        return (1, (pair, *kickers))
    return (0, tuple(ranks))


def eval_7(cards: List[Card]) -> Tuple[int, Tuple[int, ...]]:
    best = (-1, tuple())
    for comb in combinations(cards, 5):
        v = eval_5(list(comb))
        if v > best:
            best = v
    return best


def get_best_hand(cards: List[Card]) -> Tuple[int, List[str]]:
    """返回 (分类ID, 最佳5张牌字符串列表)"""
    best_val = (-1, tuple())
    best_cards = []
    for comb in combinations(cards, 5):
        v = eval_5(list(comb))
        if v > best_val:
            best_val = v
            best_cards = [str(c) for c in comb]
    return best_val[0], best_cards


# ----------------------------
# Poker engine
# ----------------------------
class Street(str, Enum):
    PREFLOP = "PREFLOP"
    FLOP = "FLOP"
    TURN = "TURN"
    RIVER = "RIVER"
    SHOWDOWN = "SHOWDOWN"
    HAND_OVER = "HAND_OVER"


class ActType(str, Enum):
    FOLD = "FOLD"
    CHECK = "CHECK"
    CALL = "CALL"
    BET = "BET"
    RAISE = "RAISE"
    ALL_IN = "ALL_IN"


@dataclass
class Action:
    type: ActType
    to_amount: Optional[int] = None  # for BET/RAISE: player's contributed_round becomes this
    min_to: Optional[int] = None
    max_to: Optional[int] = None

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Action":
        t = ActType(d["type"])
        return Action(type=t, to_amount=d.get("to_amount"))


@dataclass
class Seat:
    player_id: str
    name: str
    stack: int
    in_hand: bool = True
    all_in: bool = False
    contributed_total: int = 0
    contributed_round: int = 0
    hole: List[Card] = None
    acted_this_round: bool = False
    total_profit: int = 0

    def reset_for_hand(self):
        self.in_hand = self.stack > 0
        self.all_in = False
        self.contributed_total = 0
        self.contributed_round = 0
        self.hole = []
        self.acted_this_round = False


def next_seat(i: int, n: int) -> int:
    return (i + 1) % n


def iter_seats_from(start: int, n: int):
    for k in range(n):
        yield (start + k) % n


@dataclass
class GameState:
    seats: List[Seat]
    button: int = 0
    sb: int = 1
    bb: int = 2
    street: Street = Street.HAND_OVER

    # hand state
    deck: Optional[Deck] = None
    board: List[Card] = None
    pot_total: int = 0

    # betting round state
    current_bet: int = 0          # max contributed_round among active this street
    min_raise: int = 0            # minimum increment size (simplified)
    to_act: Optional[int] = None  # seat index
    first_to_act: Optional[int] = None # first player who can act this street
    last_agg: Optional[int] = None  # last aggressor seat index (raise/bet)
    action_history: List[Dict[str, Any]] = None
    sb_pos: Optional[int] = None
    bb_pos: Optional[int] = None

    def alive_indices(self) -> List[int]:
        return [i for i, s in enumerate(self.seats) if s.in_hand]

    def active_can_act(self) -> List[int]:
        return [i for i, s in enumerate(self.seats) if s.in_hand and not s.all_in]

    def reset_round(self):
        for s in self.seats:
            s.contributed_round = 0
            s.acted_this_round = False
        self.current_bet = 0
        self.min_raise = self.bb  # simplified: min bet/raise baseline
        self.to_act = None
        self.last_agg = None

    def to_public_state(self) -> Dict[str, Any]:
        show_all_holes = self.street in (Street.SHOWDOWN, Street.HAND_OVER)
        
        seats_info = []
        for i, s in enumerate(self.seats):
            info = {
                "player_id": s.player_id,
                "name": s.name,
                "stack": s.stack,
                "in_hand": s.in_hand,
                "all_in": s.all_in,
                "contributed_total": s.contributed_total,
                "contributed_round": s.contributed_round,
                "total_profit": s.total_profit,
            }
            
            # 只有在摊牌或结束后，且玩家未弃牌时，才对所有人公开底牌
            if s.hole:
                if show_all_holes and s.in_hand:
                    info["hole"] = [str(c) for c in s.hole]
                    # 计算最佳五张和牌型名
                    seven = s.hole + (self.board or [])
                    if len(seven) >= 5:
                        cat_id, best_5 = get_best_hand(seven)
                        info["hand_type"] = HAND_CATEGORIES.get(cat_id, "未知")
                        info["best_5"] = best_5
                else:
                    # 否则只显示背面
                    info["hole"] = ["??", "??"] if len(s.hole) == 2 else []
            else:
                info["hole"] = []
            
            seats_info.append(info)

        return {
            "button": self.button,
            "sb": self.sb,
            "bb": self.bb,
            "street": self.street.value,
            "board": [str(c) for c in (self.board or [])],
            "pot_total": self.pot_total,
            "current_bet": self.current_bet,
            "min_raise": self.min_raise,
            "to_act": self.to_act,
            "sb_pos": self.sb_pos,
            "bb_pos": self.bb_pos,
            "seats": seats_info,
            "history": self.action_history[-40:] if self.action_history else [],
        }


class HoldemEngine:
    def __init__(self, state: GameState, rng_seed: int = 1):
        self.st = state
        self.rng = random.Random(rng_seed)

    def start_hand(self):
        # Update seed for every hand to ensure randomness
        self.rng = random.Random(random.randint(0, 1000000))
        
        # reset seats
        for s in self.st.seats:
            s.reset_for_hand()

        alive = self.st.alive_indices()
        if len(alive) < 2:
            self.st.street = Street.HAND_OVER
            return

        self.st.street = Street.PREFLOP
        self.st.deck = Deck(self.rng)
        self.st.board = []
        self.st.pot_total = 0
        self.st.action_history = []

        # deal hole
        for i in alive:
            self.st.seats[i].hole = self.st.deck.deal(2)

        # post blinds
        self._post_blinds()

        # init preflop betting
        self._start_betting_round_preflop()

    def _post_blinds(self):
        alive = self.st.alive_indices()
        if len(alive) == 2:
            sb_i = self._first_alive_from(self.st.button)
            bb_i = self._next_alive(sb_i)
        else:
            sb_i = self._next_alive(self.st.button)
            bb_i = self._next_alive(sb_i)

        self.st.sb_pos = sb_i
        self.st.bb_pos = bb_i

        self._commit(sb_i, min(self.st.sb, self.st.seats[sb_i].stack), blind=True, label="SB")
        self._commit(bb_i, min(self.st.bb, self.st.seats[bb_i].stack), blind=True, label="BB")

        # set current_bet = BB contributed_round
        self.st.current_bet = self.st.seats[bb_i].contributed_round
        self.st.min_raise = self.st.bb  # simplified

    def _start_betting_round_preflop(self):
        alive = self.st.alive_indices()
        if len(alive) == 2:
            sb_i = self._first_alive_from(self.st.button)
            bb_i = self._next_alive(sb_i)
            self.st.to_act = sb_i
        else:
            sb_i = self._next_alive(self.st.button)
            bb_i = self._next_alive(sb_i)
            self.st.to_act = self._next_alive(bb_i)
        
        self.st.first_to_act = self.st.to_act
        self.st.last_agg = bb_i  # BB is last aggressor for "closing" preflop

    def _start_betting_round_postflop(self):
        # postflop first to act: first alive left of button who is NOT all-in
        self.st.to_act = self._next_to_act(self.st.button)
        self.st.first_to_act = self.st.to_act
        self.st.last_agg = None

    def legal_actions(self, seat_i: int) -> List[Action]:
        s = self.st.seats[seat_i]
        if not s.in_hand or s.all_in or self.st.to_act != seat_i:
            return []
        to_call = self.st.current_bet - s.contributed_round
        acts: List[Action] = [Action(ActType.FOLD)]

        if to_call <= 0:
            acts.append(Action(ActType.CHECK))
            # can bet/raise if has chips
            if s.stack > 0:
                if self.st.current_bet == 0:
                    min_to = s.contributed_round + max(self.st.bb, 1)
                    max_to = s.contributed_round + s.stack
                    if max_to >= min_to:
                        acts.append(Action(ActType.BET, min_to=min_to, max_to=max_to))
                else:
                    # This happens for BB in preflop when everyone calls
                    min_to = self.st.current_bet + self.st.min_raise
                    max_to = s.contributed_round + s.stack
                    if max_to >= min_to:
                        acts.append(Action(ActType.RAISE, min_to=min_to, max_to=max_to))
        else:
            # can call (possibly all-in)
            acts.append(Action(ActType.CALL, to_amount=min(s.stack, to_call)))
            # can raise if has extra chips beyond call
            if s.stack > to_call:
                min_to = self.st.current_bet + self.st.min_raise
                max_to = s.contributed_round + s.stack
                if max_to >= min_to:
                    acts.append(Action(ActType.RAISE, min_to=min_to, max_to=max_to))

        # Always add ALL_IN if they have stack
        if s.stack > 0:
            acts.append(Action(ActType.ALL_IN, to_amount=s.contributed_round + s.stack))
            
        return acts

    def apply_action(self, seat_i: int, action: Action):
        if self.st.to_act != seat_i:
            raise ValueError("Not your turn")
        s = self.st.seats[seat_i]
        if not s.in_hand or s.all_in:
            raise ValueError("Cannot act")

        orig_type = action.type
        if action.type == ActType.ALL_IN:
            action.to_amount = s.contributed_round + s.stack
            if self.st.current_bet == 0:
                action.type = ActType.BET
            else:
                if action.to_amount <= self.st.current_bet:
                    action.type = ActType.CALL
                else:
                    action.type = ActType.RAISE

        to_call = self.st.current_bet - s.contributed_round

        def log(extra=None):
            self.st.action_history.append({
                "street": self.st.street.value,
                "seat": seat_i,
                "player_id": s.player_id,
                "name": s.name,
                "action": orig_type.value, # Use original type for logging
                "to_amount": action.to_amount,
                "extra": extra,
            })

        s.acted_this_round = True

        if action.type == ActType.FOLD:
            s.in_hand = False
            log()
        elif action.type == ActType.CHECK:
            if to_call != 0:
                raise ValueError("Cannot check when facing a bet")
            log()
        elif action.type == ActType.CALL:
            if to_call <= 0:
                # treat as check
                log({"treated_as": "CHECK"})
            else:
                amt = min(to_call, s.stack)
                self._commit(seat_i, amt)
                log({"called": amt, "to_call": to_call})
        elif action.type in (ActType.BET, ActType.RAISE):
            if action.to_amount is None:
                raise ValueError("Missing to_amount")
            target = action.to_amount
            if target <= s.contributed_round:
                raise ValueError("to_amount must increase your contributed_round")
            need = target - s.contributed_round
            if need > s.stack:
                raise ValueError("Not enough chips")
            # If BET, must be no current bet
            if action.type == ActType.BET and self.st.current_bet != 0:
                raise ValueError("Cannot bet after a bet exists; use RAISE")
            # If RAISE, must face a bet
            if action.type == ActType.RAISE and self.st.current_bet == 0:
                raise ValueError("Cannot raise when no bet; use BET")

            # enforce min-raise (simplified, allow all-in smaller as "call-like")
            new_bet = target
            raise_size = new_bet - self.st.current_bet
            is_all_in = (need == s.stack)
            if raise_size < self.st.min_raise and not is_all_in:
                raise ValueError("Raise too small")

            self._commit(seat_i, need)
            # update betting state
            if new_bet > self.st.current_bet:
                self.st.min_raise = max(self.st.min_raise, raise_size)
                self.st.current_bet = new_bet
                self.st.last_agg = seat_i
                # reset other's acted status because the bet has increased
                for i, other in enumerate(self.st.seats):
                    if i != seat_i:
                        other.acted_this_round = False

            log({"need": need, "new_current_bet": self.st.current_bet})
        else:
            raise ValueError("Unknown action")

        # advance game
        self._advance_after_action()

    def _commit(self, seat_i: int, amt: int, blind=False, label=None):
        s = self.st.seats[seat_i]
        amt = max(0, min(amt, s.stack))
        s.stack -= amt
        s.total_profit -= amt
        s.contributed_total += amt
        s.contributed_round += amt
        self.st.pot_total += amt
        if s.stack == 0 and s.in_hand:
            s.all_in = True
        if blind:
            self.st.action_history.append({
                "street": self.st.street.value,
                "seat": seat_i,
                "player_id": s.player_id,
                "name": s.name,
                "action": label or "BLIND",
                "to_amount": s.contributed_round,
            })

    def _advance_after_action(self):
        # if only one remains -> award pot
        alive = self.st.alive_indices()
        if len(alive) == 1:
            winner = alive[0]
            win_amount = self.st.pot_total
            self.st.seats[winner].stack += win_amount
            self.st.seats[winner].total_profit += win_amount
            
            results = []
            for i, s in enumerate(self.st.seats):
                if s.contributed_total > 0:
                    results.append({
                        "seat": i,
                        "name": s.name,
                        "won": win_amount if i == winner else 0,
                        "lost": s.contributed_total,
                    })

            self.st.action_history.append({
                "street": self.st.street.value,
                "action": "WIN_NO_SHOWDOWN",
                "winner_seat": winner,
                "winner": self.st.seats[winner].name,
                "amount": win_amount,
                "results": results
            })
            self.st.street = Street.HAND_OVER
            self.st.to_act = None
            return

        # move to next player
        self.st.to_act = self._next_to_act(self.st.to_act)

        # if betting round complete -> move street
        if self._betting_round_complete():
            self._go_next_street()

    def _betting_round_complete(self) -> bool:
        # 1. Check if everyone in_hand has had a chance to act and matched the current_bet
        alive = self.st.alive_indices()
        for i in alive:
            s = self.st.seats[i]
            if s.all_in:
                continue
            if not s.acted_this_round or s.contributed_round != self.st.current_bet:
                return False
        
        # 2. If we reach here, either everyone matched or everyone is all-in.
        # But we also need to make sure that if anyone CAN still act, we don't end prematurely
        # if they haven't had their turn yet. (Actually the acted_this_round handles this).
        return True

    def _go_next_street(self):
        # reset betting round
        self.st.reset_round()

        if self.st.street == Street.PREFLOP:
            self.st.street = Street.FLOP
            self.st.board += self.st.deck.deal(3)
            self._start_betting_round_postflop()
        elif self.st.street == Street.FLOP:
            self.st.street = Street.TURN
            self.st.board += self.st.deck.deal(1)
            self._start_betting_round_postflop()
        elif self.st.street == Street.TURN:
            self.st.street = Street.RIVER
            self.st.board += self.st.deck.deal(1)
            self._start_betting_round_postflop()
        elif self.st.street == Street.RIVER:
            self.st.street = Street.SHOWDOWN
            self._showdown()
            return # _showdown sets HAND_OVER
        else:
            self.st.street = Street.HAND_OVER
            self.st.to_act = None
            return

        # If everyone is all-in, this street might be complete immediately
        if self._betting_round_complete():
            self._go_next_street()

    def _showdown(self):
        alive = self.st.alive_indices()
        # build side pots
        pots = build_side_pots(self.st.seats)
        ranks = {}
        hand_names = {}
        for i in alive:
            seven = self.st.seats[i].hole + self.st.board
            v = eval_7(seven)
            ranks[i] = v
            hand_names[i] = HAND_CATEGORIES.get(v[0], "未知")

        payouts = []
        won_amounts = {i: 0 for i in range(len(self.st.seats))}
        
        for pot_amount, eligible in pots:
            elig_alive = [i for i in eligible if self.st.seats[i].in_hand]
            if not elig_alive:
                continue
            best = max(ranks[i] for i in elig_alive)
            winners = [i for i in elig_alive if ranks[i] == best]
            share = pot_amount // len(winners)
            rem = pot_amount % len(winners)
            for w in winners:
                self.st.seats[w].stack += share
                self.st.seats[w].total_profit += share
                won_amounts[w] += share
            # remainder: give to winners starting from left of button
            if rem:
                order = list(iter_seats_from(next_seat(self.st.button, len(self.st.seats)), len(self.st.seats)))
                ordered_winners = [i for i in order if i in winners]
                for k in range(rem):
                    win_idx = ordered_winners[k % len(ordered_winners)]
                    self.st.seats[win_idx].stack += 1
                    self.st.seats[win_idx].total_profit += 1
                    won_amounts[win_idx] += 1

            payouts.append({
                "pot": pot_amount,
                "eligible": elig_alive,
                "winners": winners,
            })

        # Calculate profit/loss for everyone
        results = []
        for i, s in enumerate(self.st.seats):
            if s.contributed_total > 0:
                results.append({
                    "seat": i,
                    "name": s.name,
                    "won": won_amounts.get(i, 0),
                    "lost": s.contributed_total,
                    "hand_type": hand_names.get(i, "弃牌"),
                    "hole": [str(c) for c in s.hole] if s.hole else []
                })

        self.st.action_history.append({
            "street": self.st.street.value,
            "action": "SHOWDOWN",
            "results": results,
            "board": [str(c) for c in self.st.board],
        })
        self.st.street = Street.HAND_OVER
        self.st.to_act = None

    def _next_alive(self, from_i: int) -> int:
        n = len(self.st.seats)
        j = next_seat(from_i, n)
        for _ in range(n):
            if self.st.seats[j].in_hand:
                return j
            j = next_seat(j, n)
        return from_i

    def _first_alive_from(self, from_i: int) -> int:
        n = len(self.st.seats)
        j = from_i
        for _ in range(n):
            if self.st.seats[j].in_hand:
                return j
            j = next_seat(j, n)
        return from_i

    def _next_to_act(self, from_i: int) -> int:
        n = len(self.st.seats)
        j = next_seat(from_i, n)
        for _ in range(n):
            s = self.st.seats[j]
            if s.in_hand and not s.all_in:
                return j
            j = next_seat(j, n)
        return from_i


def build_side_pots(seats: List[Seat]) -> List[Tuple[int, List[int]]]:
    # returns list of (pot_amount, eligible_indices)
    contribs = [(i, s.contributed_total) for i, s in enumerate(seats) if s.contributed_total > 0]
    if not contribs:
        return []
    levels = sorted(set(c for _, c in contribs))
    pots = []
    prev = 0
    for lvl in levels:
        delta = lvl - prev
        if delta <= 0:
            continue
        contributors = [i for i, s in enumerate(seats) if s.contributed_total >= lvl]
        pot_amount = delta * len(contributors)
        eligible = [i for i in contributors if seats[i].in_hand]  # only non-folded can win
        pots.append((pot_amount, eligible))
        prev = lvl
    return pots


# ----------------------------
# Rooms + WebSocket hub
# ----------------------------
# ----------------------------
# AI Helpers
# ----------------------------
def round_to_10(n: int) -> int:
    """取最近的10倍数"""
    return round(n / 10) * 10

def get_smart_bet_amount(min_to: int, max_to: int, pot_total: int) -> int:
    """随机选择一个池底比例档位并取整"""
    gears = [0.33, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0]
    gear = random.choice(gears)
    # 计算相对于池底的加注额
    target = min_to + (pot_total * gear)
    rounded = round_to_10(int(target))
    return max(min_to, min(max_to, rounded))

def estimate_win_rate(hole_cards: List[Card], board_cards: List[Card], num_opponents: int, iterations: int = 100) -> float:
    """使用蒙特卡洛模拟估算胜率"""
    # 移除已知的牌
    used_cards = set(hole_cards) | set(board_cards)
    full_deck = [Card(r, s) for r in RANKS for s in SUITS]
    remaining_deck = [c for c in full_deck if c not in used_cards]
    
    wins = 0
    for _ in range(iterations):
        random.shuffle(remaining_deck)
        # 补全公共牌
        needed_board = 5 - len(board_cards)
        sim_board = board_cards + remaining_deck[:needed_board]
        
        # 给对手发牌
        ptr = needed_board
        my_rank = eval_7(hole_cards + sim_board)
        
        is_win = True
        for _ in range(num_opponents):
            opp_hole = remaining_deck[ptr : ptr + 2]
            ptr += 2
            opp_rank = eval_7(opp_hole + sim_board)
            if opp_rank > my_rank:
                is_win = False
                break
        
        if is_win:
            wins += 1
            
    return wins / iterations

def preflop_strength(hole: List[Card]) -> float:
    """简单的翻牌前牌力评估 (0-1)"""
    r1, r2 = sorted([hole[0].r, hole[1].r], reverse=True)
    suited = hole[0].s == hole[1].s
    pair = r1 == r2
    
    # 基础分
    score = r1 * 2 + r2
    if pair: score += 20
    if suited: score += 10
    if r1 - r2 == 1: score += 5 # 连张
    
    # 归一化 (最高 AA 约 60分, 最低 2-7 约 8分)
    return min(1.0, score / 50.0)

class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def act(self, state: Dict[str, Any]) -> Action:
        raise NotImplementedError

class SimpleAgent(BaseAgent):
    """简单机器人：基于固定概率分布，比较随意，且容易上头"""
    def act(self, state: Dict[str, Any]) -> Action:
        legal_actions = state["legal_actions"]
        if not legal_actions: return Action(ActType.FOLD)
        acts = {a["type"]: a for a in legal_actions}
        check_act, call_act = acts.get(ActType.CHECK.value), acts.get(ActType.CALL.value)
        raise_act, bet_act = acts.get(ActType.RAISE.value), acts.get(ActType.BET.value)
        
        # 30% 上头概率：无脑跟注
        if random.random() < 0.30:
            if call_act: return Action(ActType.CALL, to_amount=call_act["to_amount"])
            if check_act: return Action(ActType.CHECK)

        r = random.random()
        if check_act:
            target = bet_act or raise_act
            if r < 0.2 and target:
                to_amt = get_smart_bet_amount(target["min_to"], target["max_to"], state["public"]["pot_total"])
                return Action(ActType(target["type"]), to_amount=to_amt)
            return Action(ActType.CHECK)
        if call_act:
            if r < 0.7: return Action(ActType.CALL, to_amount=call_act["to_amount"])
            if r < 0.85 and raise_act:
                to_amt = get_smart_bet_amount(raise_act["min_to"], raise_act["max_to"], state["public"]["pot_total"])
                return Action(ActType.RAISE, to_amount=to_amt)
            return Action(ActType.FOLD)
        return Action(ActType.FOLD)

class MediumAgent(BaseAgent):
    """中等机器人：基于胜率评估，相对理性，具备基础的诈唬能力"""
    def act(self, state: Dict[str, Any]) -> Action:
        legal_actions = state["legal_actions"]
        if not legal_actions: return Action(ActType.FOLD)
        acts = {a["type"]: a for a in legal_actions}
        
        check_act, call_act = acts.get(ActType.CHECK.value), acts.get(ActType.CALL.value)
        raise_act, bet_act = acts.get(ActType.RAISE.value), acts.get(ActType.BET.value)

        # 1. 基础概率计算
        hole = [Card.from_str(c) for c in state["your_hole"]]
        board = [Card.from_str(c) for c in state["public"]["board"]]
        street = state["public"]["street"]
        opponents = max(1, sum(1 for s in state["public"]["seats"] if s["in_hand"] and s["player_id"] != state["player_id"]))
        win_rate = preflop_strength(hole) if street == "PREFLOP" else estimate_win_rate(hole, board, opponents, iterations=200)

        r = random.random()

        # 2. 状态逻辑：上头 (10% 概率无脑跟注)
        if r < 0.10:
            if call_act: return Action(ActType.CALL, to_amount=call_act["to_amount"])
            if check_act: return Action(ActType.CHECK)

        # 3. 状态逻辑：诈唬 (Bluffing)
        target = raise_act or bet_act
        if target:
            # 3.1 纯诈唬 (Pure Bluff): 没牌也要打，5% 概率
            if win_rate < 0.20 and r < 0.15: # 0.10(tilt) + 0.05(bluff)
                to_amt = get_smart_bet_amount(target["min_to"], target["max_to"], state["public"]["pot_total"])
                return Action(ActType(target["type"]), to_amount=to_amt)
            
            # 3.2 半诈唬 (Semi-Bluff): 牌力一般但试图施压，10% 概率
            if 0.20 <= win_rate < 0.40 and r < 0.20: # 0.10(tilt) + 0.10(bluff)
                to_amt = get_smart_bet_amount(target["min_to"], target["max_to"], state["public"]["pot_total"])
                return Action(ActType(target["type"]), to_amount=to_amt)

        # 4. 常规理性逻辑
        if win_rate > 0.75:
            if target:
                to_amt = get_smart_bet_amount(target["min_to"], target["max_to"], state["public"]["pot_total"])
                return Action(ActType(target["type"]), to_amount=to_amt)
            return Action(ActType.CALL if call_act else ActType.CHECK)
        elif win_rate > 0.4:
            return Action(ActType.CALL, to_amount=call_act["to_amount"]) if call_act else Action(ActType.CHECK)
        elif win_rate > 0.2:
            if check_act: return Action(ActType.CHECK)
            if call_act and call_act["to_amount"] < 100: return Action(ActType.CALL, to_amount=call_act["to_amount"])
            return Action(ActType.FOLD)
        else:
            if check_act: return Action(ActType.CHECK)
            return Action(ActType.FOLD)

# ----------------------------
# RL Architecture
# ----------------------------
class FeatureExtractor:
    """将游戏状态转换为神经网络可读的张量 (List[float])"""
    @staticmethod
    def extract(state: Dict[str, Any]) -> List[float]:
        features = []
        
        # 1. 阶段信息 (One-hot 编码)
        streets = ["PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]
        current_street = state["public"]["street"]
        for s in streets:
            features.append(1.0 if current_street == s else 0.0)
            
        # 2. 牌面信息 (归一化到 0-1)
        # 手牌
        hole = [Card.from_str(c) for c in state["your_hole"]]
        for i in range(2):
            if i < len(hole):
                features.append(hole[i].r / 14.0)
                features.append({"s":0.1, "h":0.4, "d":0.7, "c":1.0}[hole[i].s])
            else:
                features.extend([0.0, 0.0])
        
        # 公共牌 (最多5张)
        board = [Card.from_str(c) for c in state["public"]["board"]]
        for i in range(5):
            if i < len(board):
                features.append(board[i].r / 14.0)
                features.append({"s":0.1, "h":0.4, "d":0.7, "c":1.0}[board[i].s])
            else:
                features.extend([0.0, 0.0])
                
        # 3. 筹码与底池 (以大盲 BB 为基准归一化)
        bb = state["public"]["bb"]
        features.append(min(10.0, state["public"]["pot_total"] / (bb * 100.0))) # 底池大小
        
        # 自己的筹码和投入
        my_seat_idx = state["your_seat"]
        my_seat = state["public"]["seats"][my_seat_idx]
        features.append(min(10.0, my_seat["stack"] / (bb * 100.0)))
        features.append(min(2.0, my_seat["contributed_round"] / (bb * 20.0)))
        
        # 需要跟注的额度
        to_call = state["public"]["current_bet"] - my_seat["contributed_round"]
        features.append(min(2.0, to_call / (bb * 20.0)))
        
        # 4. 玩家状态
        # 活跃人数比例
        active_players = sum(1 for s in state["public"]["seats"] if s["in_hand"])
        features.append(active_players / 9.0)
        
        # 相对位置 (Button=0)
        btn = state["public"]["button"]
        pos_relative = (my_seat_idx - btn + 9) % 9
        features.append(pos_relative / 9.0)
        
        return features

class RLAgent(BaseAgent):
    """高级机器人：基于强化学习模型推理"""
    def __init__(self, name: str):
        super().__init__(name)
        self.model = None
        self.device = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            # 这里需要引用 train_rl 里的网络结构
            # 简单起见，我们直接在类内定义或者通过这种方式
            from train_rl import PokerPolicyNet, map_idx_to_action
            
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.model = PokerPolicyNet().to(self.device)
            
            model_path = "models/poker_rl_latest.pth"
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                self.model.eval()
                # print(f"RL Agent {self.name} loaded model from {model_path}")
        except Exception as e:
            # print(f"RL Agent {self.name} failed to load model: {e}")
            pass

    def act(self, state: Dict[str, Any]) -> Action:
        legal_actions = state["legal_actions"]
        if not legal_actions: return Action(ActType.FOLD)
        
        if self.model:
            try:
                import torch
                from train_rl import map_idx_to_action
                obs = FeatureExtractor.extract(state)
                obs_t = torch.FloatTensor(obs).to(self.device)
                
                with torch.no_grad():
                    probs, _ = self.model(obs_t)
                    # probs 形状为 [1, 6]，取第一行后找最大值
                    action_idx = torch.argmax(probs[0]).item()
                
                return map_idx_to_action(action_idx, legal_actions, state)
            except Exception:
                pass
        
        # 兜底：如果模型未加载或报错，使用中级机器人逻辑
        return MediumAgent(self.name).act(state)

class Room:
    def __init__(self, room_id: str, sb: int = 10, bb: int = 20, seed: int = 42):
        self.room_id = room_id
        self.lock = asyncio.Lock()
        self.connections: Dict[str, WebSocket] = {}
        self.player_seat: Dict[str, int] = {}
        self.ai_agents: Dict[str, BaseAgent] = {} # player_id -> Agent
        self.ready: set[str] = set()
        self.max_seats = 9 # 设置最大人数为 9

        self.state = GameState(seats=[], sb=sb, bb=bb, button=0)
        self.engine = HoldemEngine(self.state, rng_seed=seed)

    def add_player(self, player_id: str, name: str, stack: int = 2000, is_ai: bool = False, level: str = "simple") -> int:
        if len(self.state.seats) >= self.max_seats:
            raise ValueError("房间已满")
            
        # assign seat
        seat_idx = len(self.state.seats)
        self.state.seats.append(Seat(player_id=player_id, name=name, stack=stack, hole=[]))
        self.player_seat[player_id] = seat_idx
        if is_ai:
            if level == "hard":
                self.ai_agents[player_id] = RLAgent(name)
            elif level == "medium":
                self.ai_agents[player_id] = MediumAgent(name)
            else:
                self.ai_agents[player_id] = SimpleAgent(name)
            self.ready.add(player_id) # AI always ready
        return seat_idx

    async def remove_player(self, player_id: str, reason: Optional[str] = None):
        # 移除准备状态
        self.ready.discard(player_id)
        
        # 无论什么阶段，离席都尝试触发总结发送
        # 如果还在游戏中，_real_remove_from_seats 会处理逻辑
        await self._real_remove_from_seats(player_id, reason=reason)
            
        # 最后才移除连接
        self.connections.pop(player_id, None)

    async def _real_remove_from_seats(self, player_id: str, reason: Optional[str] = None):
        if player_id in self.player_seat:
            idx = self.player_seat[player_id]
            s = self.state.seats[idx]
            is_ai = player_id in self.ai_agents
            
            # 生成结算总结数据
            summary = [
                {
                    "name": seat.name,
                    "profit": seat.total_profit,
                    "is_ai": seat.player_id in self.ai_agents
                }
                for seat in self.state.seats
            ]

            # 记录离席统计
            profit_str = ("+" if s.total_profit > 0 else "") + str(s.total_profit)
            role_desc = "机器人" if is_ai else "玩家"
            reason_str = f"{reason}，" if reason else ""
            self.state.action_history.append({
                "street": self.state.street.value,
                "action": "SYSTEM",
                "name": "系统",
                "extra": f"{role_desc} {s.name} {reason_str}离开了房间 (本场盈亏: {profit_str})"
            })

            # 如果离席的是人类玩家，且连接还存在，发送总结消息
            if not is_ai:
                ws = self.connections.get(player_id)
                if ws:
                    try:
                        # 直接 await 发送，确保客户端在连接关闭前收到
                        # 如果是连接断开导致的离席，这里会报错，我们需要捕获它
                        await send_json(ws, {
                            "type": "game_summary",
                            "summary": summary
                        })
                    except Exception:
                        # 忽略发送失败（例如连接已关闭的情况）
                        pass

            self.state.seats.pop(idx)
            # 重新映射
            self.player_seat = {s.player_id: i for i, s in enumerate(self.state.seats)}
            self.ai_agents.pop(player_id, None)
            
            # 检查是否还有人类玩家
            humans_left = any(seat.player_id for seat in self.state.seats if seat.player_id not in self.ai_agents)
            if not humans_left:
                print(f"Room {self.room_id}: No humans left, resetting room.")
                self._reset_room_fully()
            
            # 调整 button 位置
            if len(self.state.seats) > 0:
                self.state.button = self.state.button % len(self.state.seats)
            else:
                self.state.button = 0

    def _reset_room_fully(self):
        """完全重置房间状态"""
        self.state.seats = []
        self.player_seat = {}
        self.ai_agents = {}
        self.ready = set()
        self.state.button = 0
        self.state.street = Street.HAND_OVER
        self.state.pot_total = 0
        self.state.board = []
        self.state.action_history = []

    async def handle_broke_players(self):
        """检查并处理筹码用尽的玩家"""
        to_remove = []
        for s in self.state.seats:
            if s.stack <= 0:
                pid = s.player_id
                if pid in self.ai_agents:
                    # AI 逻辑: 80% 几率贷款，20% 几率离席
                    if random.random() < 0.8:
                        s.stack = 2000
                        self.state.action_history.append({
                            "street": self.state.street.value,
                            "action": "SYSTEM",
                            "name": "系统",
                            "extra": f"机器人 {s.name} 申请了 2000 贷款"
                        })
                    else:
                        to_remove.append(pid)
                else:
                    # 人类玩家: 发送贷款询问
                    ws = self.connections.get(pid)
                    if ws:
                        # 只有当还没有记录这个系统消息时才记录，避免重复
                        msg_exists = any(h.get("extra") == f"玩家 {s.name} 筹码耗尽，正在询问是否贷款" for h in self.state.action_history[-5:])
                        if not msg_exists:
                            self.state.action_history.append({
                                "street": self.state.street.value,
                                "action": "SYSTEM",
                                "name": "系统",
                                "extra": f"玩家 {s.name} 筹码耗尽，正在询问是否贷款"
                            })
                        
                        await send_json(ws, {
                            "type": "loan_request",
                            "message": "您的筹码已用尽，是否申请贷款 2000 筹码继续游戏？",
                            "amount": 2000
                        })
        
        for pid in to_remove:
            await self._real_remove_from_seats(pid, reason="筹码耗尽")

    def can_start(self) -> bool:
        # 存活且在座的人
        alive_seats = [s for s in self.state.seats if s.stack > 0]
        if len(alive_seats) < 2:
            return False
            
        # 必须所有人类玩家都点击了准备
        for s in self.state.seats:
            if s.player_id not in self.ai_agents and s.player_id not in self.ready:
                return False
                
        return self.state.street == Street.HAND_OVER

    def rotate_button(self):
        n = len(self.state.seats)
        if n == 0:
            return
        idx = (self.state.button + 1) % n
        for _ in range(n):
            if self.state.seats[idx].stack > 0:
                self.state.button = idx
                break
            idx = (idx + 1) % n

    async def check_ai_turn(self):
        """检查当前是否轮到 AI 行动，如果是则执行动作"""
        while self.state.to_act is not None:
            seat_i = self.state.to_act
            player_id = self.state.seats[seat_i].player_id
            agent = self.ai_agents.get(player_id)
            
            if agent is None:
                break # 轮到真人玩家了
                
            # 模拟思考时间
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # 获取当前 AI 的私有状态进行决策
            private_state = make_private_state(self, player_id)
            action = agent.act(private_state)
            
            # 执行动作
            try:
                self.engine.apply_action(seat_i, action)
                if self.state.street == Street.HAND_OVER:
                    await self.handle_broke_players()
                    self.rotate_button()
                # 行动后广播新状态
                await broadcast_room(self)
            except Exception as e:
                print(f"AI Action Error: {e}")
                break


rooms: Dict[str, Room] = {}


async def send_json(ws: WebSocket, obj: Dict[str, Any]):
    await ws.send_text(json.dumps(obj, ensure_ascii=False))


def make_private_state(room: Room, player_id: str) -> Dict[str, Any]:
    st = room.state
    pub = st.to_public_state()
    seat = room.player_seat.get(player_id)
    hole = []
    if seat is not None and st.seats[seat].hole:
        hole = [str(c) for c in st.seats[seat].hole]
    # legal actions
    legal = []
    if seat is not None:
        legal = [asdict(a) for a in room.engine.legal_actions(seat)]
        # Enum to str
        for a in legal:
            a["type"] = a["type"].value if hasattr(a["type"], "value") else a["type"]
    return {
        "type": "state",
        "room": room.room_id,
        "player_id": player_id, # 添加这一行
        "your_seat": seat,
        "your_hole": hole,
        "legal_actions": legal,
        "public": pub,
        "ready": list(room.ready),
    }


async def broadcast_room(room: Room):
    for pid, ws in list(room.connections.items()):
        try:
            await send_json(ws, make_private_state(room, pid))
        except Exception:
            # ignore broken sockets
            pass


@app.websocket("/ws/{room_id}/{player_id}")
async def ws_endpoint(ws: WebSocket, room_id: str, player_id: str):
    await ws.accept()
    room = rooms.get(room_id)
    if room is None:
        room = rooms[room_id] = Room(room_id, sb=10, bb=20, seed=random.randint(0, 1000000))

    async with room.lock:
        room.connections[player_id] = ws

    try:
        # wait join message
        raw = await ws.receive_text()
        msg = json.loads(raw)
        if msg.get("type") != "join":
            await send_json(ws, {"type": "error", "message": "First message must be join"})
            return
        name = msg.get("name", player_id)
        stack = int(msg.get("stack", 2000))

        async with room.lock:
            if player_id not in room.player_seat:
                try:
                    room.add_player(player_id, name=name, stack=stack)
                except ValueError as e:
                    await send_json(ws, {"type": "error", "message": str(e)})
                    return

            await broadcast_room(room)

        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            async with room.lock:
                t = msg.get("type")

                if t == "ready":
                    # 如果准备时筹码为0，触发贷款流程
                    seat_idx = room.player_seat.get(player_id)
                    if seat_idx is not None and room.state.seats[seat_idx].stack <= 0:
                        await room.handle_broke_players()
                        await broadcast_room(room)
                        continue

                    room.ready.add(player_id)
                    # auto start if possible
                    if room.can_start():
                        room.engine.start_hand()
                    else:
                        # 如果人数够了但没开始，检查是否是因为有人筹码为0
                        # 现在会尝试自动处理贷款，所以这里只是个保底提示
                        alive = [s for s in room.state.seats if s.stack > 0]
                        if len(room.ready) >= 2 and len(alive) < 2:
                            await room.handle_broke_players()
                    await broadcast_room(room)
                    await room.check_ai_turn()

                elif t == "start":
                    # manual start
                    if room.can_start():
                        room.engine.start_hand()
                    await broadcast_room(room)
                    await room.check_ai_turn()

                elif t == "add_ai":
                    level = msg.get("level", "simple")
                    
                    # 趣味昵称池
                    simple_names = ["鱼王阿强", "提款机老王", "盲注搬运工", "底池赞助商", "全押狂魔", "只会跟注的小白", "锦标赛游客", "气氛组组长"]
                    medium_names = ["冷面计算器", "职业鲨鱼", "GTO执行者", "读牌专家", "稳健派阿福", "底池收割机", "河牌收割手", "筹码终结者"]
                    hard_names = ["赌神高进", "AlphaHoldem", "深蓝扑克", "德州死神", "底池统治者", "全知之眼", "无敌寂寞手", "终极进化体"]
                    
                    if level == "hard":
                        pool = hard_names
                    elif level == "medium":
                        pool = medium_names
                    else:
                        pool = simple_names
                        
                    ai_name = random.choice(pool) + "_" + str(random.randint(10, 99))
                    
                    ai_id = f"ai_{random.randint(1000, 9999)}"
                    try:
                        room.add_player(ai_id, name=ai_name, stack=int(msg.get("stack", 2000)), is_ai=True, level=level)
                        await broadcast_room(room)
                        # 移除自动开始逻辑，等待房主手动点准备
                    except ValueError as e:
                        await send_json(ws, {"type": "error", "message": str(e)})
                    if room.can_start():
                        room.engine.start_hand()
                        await broadcast_room(room)
                        await room.check_ai_turn()

                elif t == "update_config":
                    room.state.sb = int(msg.get("sb", room.state.sb))
                    room.state.bb = int(msg.get("bb", room.state.bb))
                    await broadcast_room(room)

                elif t == "reset_stacks":
                    stack = int(msg.get("stack", 1000))
                    for s in room.state.seats:
                        s.stack = stack
                    await broadcast_room(room)

                elif t == "act":
                    seat = room.player_seat.get(player_id)
                    if seat is None:
                        await send_json(ws, {"type": "error", "message": "No seat"})
                        continue
                    try:
                        action = Action.from_json(msg["action"])
                        room.engine.apply_action(seat, action)
                    except Exception as e:
                        await send_json(ws, {"type": "error", "message": str(e)})
                        continue

                    # if hand over, rotate button automatically (casual mode)
                    if room.state.street == Street.HAND_OVER:
                        await room.handle_broke_players()
                        room.rotate_button()
                    await broadcast_room(room)
                    await room.check_ai_turn()

                elif t == "loan_response":
                    accept = msg.get("accept", False)
                    seat_idx = room.player_seat.get(player_id)
                    if seat_idx is not None:
                        s = room.state.seats[seat_idx]
                        if accept:
                            s.stack = 2000
                            room.state.action_history.append({
                                "street": room.state.street.value,
                                "action": "SYSTEM",
                                "name": "系统",
                                "extra": f"玩家 {s.name} 申请了 2000 贷款"
                            })
                            # 贷款后自动设为准备状态，方便游戏继续
                            room.ready.add(player_id)
                            if room.can_start():
                                room.engine.start_hand()
                        else:
                            await room.remove_player(player_id, reason="拒绝贷款")
                            return
                    await broadcast_room(room)
                    await room.check_ai_turn()

                elif t == "ping":
                    await send_json(ws, {"type": "pong"})

                elif t == "leave":
                    await room.remove_player(player_id, reason="主动离席")
                    # remove_player will call _real_remove_from_seats which sends the summary
                    return

                else:
                    await send_json(ws, {"type": "error", "message": f"Unknown type: {t}"})

    except WebSocketDisconnect:
        async with room.lock:
            await room.remove_player(player_id, reason="连接断开")
            await broadcast_room(room)
    except Exception:
        async with room.lock:
            await room.remove_player(player_id, reason="系统错误")
            await broadcast_room(room)
        raise