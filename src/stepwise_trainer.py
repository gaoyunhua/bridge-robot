#!/usr/bin/env python3
"""
Stepwise Training - 完整牌局训练器
包含：训练逻辑、验证函数
"""

from __future__ import annotations

import sys, os, time
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
import torch
import torch.nn as nn

from dds_teacher import DDTableCache, _parscore_ns
from env_core import BridgeEnv, PLAYER_ORDER
from model_transformer import BridgeTransformerV2
from endplay.types import Player
from rewards import contract_score

from stepwise_utils import (
    DDSTeacher,
    hide_opponent_cards,
    format_hand, deal_log,
    print_visible_hands, print_bid_step, print_play_step,
    process_bidding_step, process_play_step,
    compute_bid_par_loss, compute_play_par_loss,
    NUM_BID, NUM_PLAY
)


class FullBoardTrainer:
    """完整牌局训练器 - 发牌→叫牌→打牌→每步 DDS 训练"""

    def __init__(
        self,
        model: nn.Module,
        teacher: DDSTeacher,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None
    ):
        self.model = model
        self.teacher = teacher
        self.optimizer = optimizer
        self.device = device if device else torch.device('cpu')

        self.total_steps = 0
        self.bid_correct = 0
        self.play_correct = 0
        self.bid_total = 0
        self.play_total = 0
        self.total_loss = 0.0

    def train_board(self, env: BridgeEnv, board_idx: int = 1) -> dict:
        """训练一个完整牌局"""
        self.model.train()
        env.reset(board_num=board_idx)
        obs_orig = env._encode_obs()
        deal = env.deal

        vul = env.vul
        dealer = env.dealer

        dd_table = DDTableCache(deal)

        board_loss = 0.0
        board_par_loss = 0.0
        n_bid = 0
        n_play = 0
        correct_bid = 0
        correct_play = 0
        n_illegal_bid = 0
        n_illegal_play = 0

        self.optimizer.zero_grad()

        while not env.done and env.phase == 'bidding':
            step_data = process_bidding_step(
                env, self.model, self.teacher, dd_table,
                deal, vul, dealer, obs_orig, self.device
            )

            loss_par = compute_bid_par_loss(
                step_data['model_act'], dd_table, env, deal, self.device
            )
            loss_step = step_data['loss_policy'] + loss_par
            loss_step.backward()

            advance_act = step_data['optimal_act'] if step_data['is_illegal_prediction'] else step_data['model_act']
            if step_data['is_illegal_prediction']:
                n_illegal_bid += 1

            with torch.no_grad():
                env.step(advance_act)
            obs_orig = env._encode_obs()

            n_bid += 1
            board_loss += loss_step.item()
            board_par_loss += loss_par.item()
            if step_data['model_act'] == step_data['optimal_act']:
                correct_bid += 1

            print_bid_step(
                env._current_player(), step_data['raw_pred_act'], step_data['is_illegal_prediction'],
                advance_act, step_data['optimal_act'],
                [step_data['legal_actions_int']],
                step_data['loss_policy'], loss_par,
                step_data['bid_values'], step_data['bid_tricks'],
                step_data['legal_actions_int'], step_data['bid_action_losses']
            )
            print_visible_hands(step_data['obs_hidden'], step_data['player_idx'])

            if env.done:
                break

        declarer = None
        declarer_idx = -1
        dummy = None
        dummy_idx = -1
        lead_player = None

        if not env.done and env.phase == 'play':
            declarer = env.bidding.declarer
            declarer_idx = PLAYER_ORDER.index(declarer)
            dummy = Player((declarer.value + 2) % 4)
            dummy_idx = (declarer_idx + 2) % 4
            lead_player = Player((declarer.value + 1) % 4)

            print(f'\n\n' + '='*80)
            print(f'出牌阶段开始')
            print(f'='*80)
            print(f'  合约: {env.bidding.contract_level}{env.bidding.contract_denom.abbr}')
            print(f'  庄家: {declarer.name} (idx={declarer_idx})')
            print(f'  明手: {dummy.name} (idx={dummy_idx})')
            print(f'  首攻家: {lead_player.name} (第一个出牌人)')
            print(f'='*80 + '\n')

        while not env.done and env.phase == 'play':
            assert env.play_state is not None
            assert declarer is not None
            assert declarer_idx >= 0
            assert dummy is not None
            assert dummy_idx >= 0
            assert lead_player is not None

            step_data = process_play_step(
                env, self.model, self.teacher, dd_table, obs_orig,
                declarer, declarer_idx, dummy, dummy_idx, lead_player, self.device
            )

            if step_data is None:
                break

            loss_par = compute_play_par_loss(dd_table, env, deal, self.device)
            loss_step = step_data['loss_policy'] + loss_par
            loss_step.backward()

            advance_act = step_data['optimal_act'] if step_data['is_illegal_prediction'] else step_data['model_act']
            if step_data['is_illegal_prediction']:
                n_illegal_play += 1

            with torch.no_grad():
                env.step(advance_act)
            obs_orig = env._encode_obs()

            n_play += 1
            board_loss += loss_step.item()
            board_par_loss += loss_par.item()
            if step_data['model_act'] == step_data['optimal_act']:
                correct_play += 1

            print_play_step(
                env._current_player(), step_data['raw_pred_act'], step_data['is_illegal_prediction'],
                advance_act, step_data['optimal_act'],
                step_data['loss_policy'], loss_par,
                step_data['legal_actions_int'], step_data['play_action_losses']
            )
            print_visible_hands(step_data['obs_hidden'], step_data['player_idx'])

        vul = env.vul
        ns_par = _parscore_ns(deal, vul, dealer)
        actual_score = 0
        if env.play_state is not None:
            contract_level = env.bidding.contract_level
            contract_denom = env.bidding.contract_denom
            declarer = env.bidding.declarer
            tricks_made = env.play_state._tricks_won_ns if declarer.value % 2 == 0 else env.play_state._tricks_won_ew
            if contract_level > 0:
                actual_score = contract_score(
                    contract_level, contract_denom,
                    tricks_made, vul, tricks_made
                )
                if declarer.value % 2 != 0:
                    actual_score = -actual_score

        board_result_gap = abs(actual_score - ns_par)
        print(f'\n  === 牌局结束 ===')
        print(f'  Par分数: {ns_par}')
        print(f'  实际得分: {actual_score}')
        print(f'  牌局损失（差距）: {board_result_gap}')
        print()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        torch.cuda.empty_cache()

        self.total_steps += n_bid + n_play
        self.bid_correct += correct_bid
        self.play_correct += correct_play
        self.bid_total += n_bid
        self.play_total += n_play
        self.total_loss += board_loss

        return {
            'board_loss': board_loss,
            'board_par_loss': board_par_loss,
            'n_steps': n_bid + n_play,
            'n_bid': n_bid,
            'n_play': n_play,
            'correct_bid': correct_bid,
            'correct_play': correct_play,
            'n_illegal_bid': n_illegal_bid,
            'n_illegal_play': n_illegal_play,
        }


def run_val_game(env, model, device, teacher):
    """跑一个验证牌局 - 模型自回归推进，对比 DDS 最优"""
    deal = env.deal
    dd_table = DDTableCache(deal)

    steps_bid = 0
    correct_bid = 0
    steps_play = 0
    correct_play = 0

    while not env.done and env.phase == 'bidding':
        current = env._current_player()
        player_idx = PLAYER_ORDER.index(current)
        obs = env._encode_obs()
        obs_hidden = hide_opponent_cards(obs, player_idx, 'bidding')
        legal = env.legal_mask()
        obs_t = torch.from_numpy(obs_hidden).float().unsqueeze(0).to(device)

        with torch.no_grad():
            bid_logits, _ = model(obs_t)
            masked = bid_logits.clone()
            legal_bool = torch.from_numpy(legal[:NUM_BID]).bool().to(device)
            masked[0, ~legal_bool] = float('-inf')
            model_action = masked.argmax(dim=-1).item()

        try:
            optimal = teacher.optimal_bid(
                dd_table, deal, env.vul, PLAYER_ORDER[0],
                env.bidding.history, current
            )
            legal_actions = np.where(env.legal_mask()[:NUM_BID])[0].tolist()
            if optimal not in legal_actions:
                optimal = 35 if 35 in legal_actions else legal_actions[0]
        except Exception:
            legal_actions = np.where(env.legal_mask()[:NUM_BID])[0].tolist()
            optimal = 35 if 35 in legal_actions else legal_actions[0]

        if model_action == optimal:
            correct_bid += 1
        steps_bid += 1

        env.step(model_action)

    while not env.done and env.phase == 'play':
        assert env.play_state is not None
        current = env._current_player()
        player_idx = PLAYER_ORDER.index(current)
        declarer = env.bidding.declarer
        declarer_idx = PLAYER_ORDER.index(declarer)
        dummy_idx = (declarer_idx + 2) % 4
        is_first_trick = (env.play_state._trick_no == 0)
        trick_cards_played = len(env.play_state._current_trick_cards)

        obs = env._encode_obs()
        obs_hidden = hide_opponent_cards(
            obs, player_idx, 'play',
            declarer_idx, dummy_idx,
            is_first_trick, trick_cards_played
        )
        legal = env.legal_mask()
        obs_t = torch.from_numpy(obs_hidden).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _, play_logits = model(obs_t)
            masked = play_logits.clone()
            legal_bool = torch.from_numpy(legal[:NUM_PLAY]).bool().to(device)
            masked[0, ~legal_bool] = float('-inf')
            model_action = masked.argmax(dim=-1).item()

        legal_actions = np.where(legal[:NUM_PLAY])[0].tolist()
        try:
            optimal = teacher.optimal_card(
                dd_table, env.bidding.contract_denom, env.bidding.declarer,
                legal_actions, current, 0
            )
        except Exception:
            optimal = legal_actions[0] if legal_actions else 0

        if model_action == optimal:
            correct_play += 1
        steps_play += 1

        env.step(model_action)

    bid_acc = correct_bid / max(steps_bid, 1) * 100
    play_acc = correct_play / max(steps_play, 1) * 100
    print(f'  叫牌: {correct_bid}/{steps_bid} ({bid_acc:.0f}%) | '
          f'出牌: {correct_play}/{steps_play} ({play_acc:.0f}%)')
