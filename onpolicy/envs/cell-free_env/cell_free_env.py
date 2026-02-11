from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import simulation_utils as sim_utils
from .config import (
    AP_HEIGHT_M,
    AP_POWER_BUDGET_W,
    PENALTY_FACTOR,
    CARRIER_BANDWIDTH_HZ,
    FAIRNESS_BETA,
    GT_MAX_DBI,
    MIN_FAIR_RATE_KBPS,
    MAX_ATTEN_DB,
    SUBCARRIER_DECAY_ALPHA,
    NUM_APS,
    NUM_SUBCARRIERS,
    NUM_USERS,
    PHI_3DB_DEG,
    RANDOM_SEED,
    SIDELOBE_ATTEN_DB,
    THETA_3DB_DEG,
    TIME_STEPS,
    USER_HEIGHT_M,
)
from .simulation_utils import (
    compute_channel_tensor,
    generate_user_paths,
    place_access_points,
)

SCENARIO_CACHE_SIZE = 8  # 预构建若干随机场景以避免每次 reset 高开销

OBS_VECTOR_SIZE = (
    NUM_APS * 2       # 所有 AP 的坐标
    + NUM_APS         # 所有 AP 的下倾角
    + NUM_USERS * 2   # st 时刻所有用户位置
    + NUM_USERS * 2   # st+1 时刻所有用户位置
    + NUM_USERS       # 所有用户速率
)
ACTION_DIM = 1 + NUM_USERS + NUM_SUBCARRIERS  # downtilt + user preference + power weights


def _normalize(val: np.ndarray) -> np.ndarray:
    return np.clip(val, 0.0, 1.0)


def make_env() -> "CellFreeEnv":
    return CellFreeEnv()

def make_multiagent_env() -> "MultiAgentCellFreeEnv":
    return MultiAgentCellFreeEnv()


@dataclass
class ScenarioState:
    user_positions: np.ndarray
    user_metric_positions: np.ndarray
    ap_metric_positions: np.ndarray
    aps: list
    channels: np.ndarray
    ap_azimuth: np.ndarray
    ap_downtilt: np.ndarray
    subcarrier_users: np.ndarray
    subcarrier_powers_dbm: np.ndarray
    subcarrier_powers_w: np.ndarray
    channel_power: np.ndarray
    gains_dbi: np.ndarray = field(default_factory=lambda: np.zeros((TIME_STEPS, NUM_USERS, NUM_APS)))
    gains_linear: np.ndarray = field(default_factory=lambda: np.ones((TIME_STEPS, NUM_USERS, NUM_APS)))


class CellFreeEnv(gym.Env):
    """Gymnasium-compatible environment for the communication scenario."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # 动作: [下倾角(0-180度), 用户偏好(NUM_USERS), 子载波功率打分(NUM_SUBCARRIERS)]
        low = np.concatenate([
            np.array([0.0], dtype=np.float32),                    # downtilt
            np.full(NUM_USERS, -np.inf, dtype=np.float32),        # user preference logits
            np.full(NUM_SUBCARRIERS, -np.inf, dtype=np.float32),  # power logits
        ])
        high = np.concatenate([
            np.array([180.0], dtype=np.float32),
            np.full(NUM_USERS, np.inf, dtype=np.float32),
            np.full(NUM_SUBCARRIERS, np.inf, dtype=np.float32),
        ])
        low = np.broadcast_to(low, (NUM_APS, ACTION_DIM))
        high = np.broadcast_to(high, (NUM_APS, ACTION_DIM))
        self.action_space = spaces.Box(
            low=low, high=high, shape=(NUM_APS, ACTION_DIM), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(NUM_APS, OBS_VECTOR_SIZE),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng(RANDOM_SEED)
        self._scenario: Optional[ScenarioState] = None
        self.current_step = 0
        self.last_user_rates = np.zeros(NUM_USERS, dtype=np.float32)
        self.last_ap_subcarrier_rates = np.zeros((NUM_APS, NUM_SUBCARRIERS), dtype=np.float32)
        self.ap_metric_positions = None
        self._scenario_cache = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._scenario = self._create_scenario_template()
        else:
            self._scenario = self._get_cached_scenario()
        self.ap_metric_positions = self._scenario.ap_metric_positions
        self.current_step = 0
        self.last_user_rates = np.zeros(NUM_USERS, dtype=np.float32)
        self.last_ap_subcarrier_rates = np.zeros((NUM_APS, NUM_SUBCARRIERS), dtype=np.float32)
        observation = self._build_observation(self.current_step)
        info = {"time_step": self.current_step}
        return observation, info

    def step(self, action: np.ndarray):
        assert self._scenario is not None, "Environment must be reset before stepping."
        if action.shape != (NUM_APS, ACTION_DIM):
            raise ValueError(f"Expected action shape {(NUM_APS, ACTION_DIM)}, got {action.shape}")

        t = self.current_step
        if t >= TIME_STEPS:
            raise RuntimeError("Episode has already terminated. Reset the environment.")

        gains_dbi, gains_linear = self._apply_actions(action, t)
        self._scenario.gains_dbi[t] = gains_dbi
        self._scenario.gains_linear[t] = gains_linear
        power_tensor = self._build_power_tensor()
        channel_power = self._scenario.channel_power[t]
        user_rates = self._compute_step_rates(channel_power, power_tensor, gains_linear)
        self.last_user_rates = user_rates.astype(np.float32)
        self._update_subcarrier_rates(user_rates)
        # self._log_ap_user_allocations(t)
        # self._log_user_rates(t, user_rates)

        # 统计低速用户数量（kbps < 1024）
        low_rate_threshold = MIN_FAIR_RATE_KBPS
        low_rate_count = int(np.sum(user_rates < low_rate_threshold))

        # 奖励：分段式公平约束
        min_rate = float(np.min(user_rates))
        mean_rate = float(np.mean(user_rates))
        # print(mean_rate)
        # reward_value = mean_rate - np.sum(np.maximum(0.0, 1024.0 - user_rates))
        reward_value = mean_rate
        # alpha = 2.0
        # reward_value = np.sum(np.power(user_rates + 1e-6, 1 - alpha) / (1 - alpha)) / 1000.0  # 转为 kbps
        # reward_value = np.sum(np.log(user_rates + 1e-6) + 0.001 * user_rates)
        # print('log', np.log(user_rates + 1e-6))
        # print('linear', 0.01 * user_rates)
        fairness_term = mean_rate

        reward = np.full((NUM_APS, 1), reward_value, dtype=np.float32)
        next_time_index = min(t + 1, TIME_STEPS - 1)
        observation = self._build_observation(next_time_index)
        self.current_step += 1
        terminated = self.current_step >= TIME_STEPS
        truncated = False
        info = {
            "time_step": t,
            "user_rates": user_rates,
            "low_rate_count": low_rate_count,
            "low_rate_threshold_kbps": low_rate_threshold,
            "min_user_rate_kbps": min_rate,
            "mean_user_rate_kbps": mean_rate,
            "fairness_term": fairness_term,
        }
        return observation, reward, terminated, truncated, info

    def _build_scenario(self) -> ScenarioState:
        # 留作兼容，内部调用 _create_scenario_template
        return self._create_scenario_template()

    def _create_scenario_template(self) -> ScenarioState:
        user_positions = generate_user_paths(NUM_USERS, self._rng, TIME_STEPS)
        user_metric_positions = user_positions.copy()

        aps, ap_metric_positions = place_access_points(self._rng)
        ap_azimuth = np.zeros(len(aps), dtype=np.float64)
        ap_downtilt = np.array([ap["downtilt"] for ap in aps], dtype=np.float64)
        subcarrier_users = np.array([ap["subcarrier_users"] for ap in aps], dtype=np.int64)
        subcarrier_powers_dbm = np.array([ap["subcarrier_powers_dbm"] for ap in aps], dtype=np.float64)
        subcarrier_powers_w = np.array([ap["subcarrier_powers_w"] for ap in aps], dtype=np.float64)

        self.ap_metric_positions = ap_metric_positions

        dummy_gains = np.ones((TIME_STEPS, NUM_USERS, NUM_APS))
        channels = compute_channel_tensor(
            user_metric_positions, self.ap_metric_positions, self._rng, dummy_gains
        )
        channel_power = np.abs(channels) ** 2
        return ScenarioState(
            user_positions=user_positions,
            user_metric_positions=user_metric_positions,
            ap_metric_positions=self.ap_metric_positions,
            aps=aps,
            channels=channels,
            ap_azimuth=ap_azimuth,
            ap_downtilt=ap_downtilt,
            subcarrier_users=subcarrier_users,
            subcarrier_powers_dbm=subcarrier_powers_dbm,
            subcarrier_powers_w=subcarrier_powers_w,
            channel_power=channel_power,
        )

    def _clone_scenario(self, template: ScenarioState) -> ScenarioState:
        return ScenarioState(
            user_positions=template.user_positions.copy(),
            user_metric_positions=template.user_metric_positions.copy(),
            ap_metric_positions=template.ap_metric_positions.copy(),
            aps=copy.deepcopy(template.aps),
            channels=template.channels.copy(),
            ap_azimuth=template.ap_azimuth.copy(),
            ap_downtilt=template.ap_downtilt.copy(),
            subcarrier_users=template.subcarrier_users.copy(),
            subcarrier_powers_dbm=template.subcarrier_powers_dbm.copy(),
            subcarrier_powers_w=template.subcarrier_powers_w.copy(),
            channel_power=template.channel_power.copy(),
            gains_dbi=np.zeros_like(template.gains_dbi),
            gains_linear=np.ones_like(template.gains_linear),
        )

    def _get_cached_scenario(self) -> ScenarioState:
        if not self._scenario_cache:
            for _ in range(SCENARIO_CACHE_SIZE):
                self._scenario_cache.append(self._create_scenario_template())
        idx = int(self._rng.integers(len(self._scenario_cache)))
        template = self._scenario_cache[idx]
        return self._clone_scenario(template)

    def _apply_actions(self, action: np.ndarray, time_index: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._scenario is not None
        downtilt_deg = np.clip(action[:, 0], 0.0, 180.0).astype(np.float64)
        user_weights = action[:, 1 : 1 + NUM_USERS].astype(np.float64)
        power_weights = action[:, 1 + NUM_USERS :].astype(np.float64)

        # 若外部未归一化，做简单比例归一（非 softmax）
        user_weights = np.maximum(user_weights, 0.0)
        user_sums = np.sum(user_weights, axis=1, keepdims=True)
        user_sums = np.where(user_sums <= 0.0, 1.0, user_sums)
        user_weights = user_weights / user_sums

        power_weights = np.maximum(power_weights, 0.0)
        power_sums = np.sum(power_weights, axis=1, keepdims=True)
        power_sums = np.where(power_sums <= 0.0, 1.0, power_sums)
        power_weights = power_weights / power_sums

        downtilt_rad = np.deg2rad(downtilt_deg)
        self._scenario.ap_azimuth = np.zeros(NUM_APS, dtype=np.float64)  # 固定朝北
        self._scenario.ap_downtilt = downtilt_rad

        gains_dbi, gains_linear = self._compute_gains_for_step(time_index)
        channel_power_t = self._scenario.channel_power[time_index]  # [users, aps, subcarriers]
        subcarrier_users = np.zeros((NUM_APS, NUM_SUBCARRIERS), dtype=np.int64)

        for m in range(NUM_APS):
            # per user gain*channel power for each subcarrier
            gain_slice = gains_linear[:, m]  # [users]
            assigned_counts = np.zeros(NUM_USERS, dtype=np.int64)
            for n in range(NUM_SUBCARRIERS):
                decay = 1.0 / (1.0 + SUBCARRIER_DECAY_ALPHA * assigned_counts)
                # 计算信道增益与功率的乘积并归一化到 [0,1]
                channel_gain_product = gain_slice * channel_power_t[:, m, n]
                channel_gain_min = np.min(channel_gain_product)
                channel_gain_max = np.max(channel_gain_product)
                channel_gain_range = channel_gain_max - channel_gain_min
                if channel_gain_range > 0:
                    normalized_product = (channel_gain_product - channel_gain_min) / channel_gain_range
                else:
                    normalized_product = np.ones_like(channel_gain_product)
                metric = user_weights[m] * normalized_product * decay
                best_user = int(np.argmax(metric))
                subcarrier_users[m, n] = best_user
                assigned_counts[best_user] += 1

        powers_w = power_weights * AP_POWER_BUDGET_W
        powers_dbm = 10 * np.log10(np.clip(powers_w, 1e-12, None)) + 30.0

        self._scenario.subcarrier_users = subcarrier_users
        self._scenario.subcarrier_powers_dbm = powers_dbm
        self._scenario.subcarrier_powers_w = powers_w

        # 保持原有字典结构的同步（便于调试/兼容）
        for m, ap in enumerate(self._scenario.aps):
            ap["subcarrier_users"] = subcarrier_users[m].tolist()
            ap["subcarrier_powers_dbm"] = powers_dbm[m].tolist()
            ap["subcarrier_powers_w"] = powers_w[m].tolist()
            ap["azimuth"] = 0.0
            ap["downtilt"] = float(downtilt_rad[m])
        return gains_dbi, gains_linear

    def _compute_step_rates(
        self,
        channel_power: np.ndarray,
        power_tensor: np.ndarray,
        gains_linear_slice: np.ndarray,
    ) -> np.ndarray:
        noise = sim_utils.NOISE_POWER_W
        bandwidth_per_sub = CARRIER_BANDWIDTH_HZ / NUM_SUBCARRIERS

        # useful: [users, subcarriers]
        useful = np.sum(power_tensor * gains_linear_slice[:, :, None] * channel_power, axis=1)

        # total received power per (user, subcarrier): sum over all users' allocations,
        # but using victim user's channel/gain (same semantics as旧循环).
        # Sum power across all src users per (ap, subcarrier)
        power_sum = np.sum(power_tensor, axis=0)  # [aps, subcarriers]
        total_rx = np.sum(power_sum[None, :, :] * gains_linear_slice[:, :, None] * channel_power, axis=1)

        interference = total_rx - useful  # exclude own useful part
        sinr = useful / (noise + interference + 1e-15)
        rates = bandwidth_per_sub * np.log2(1.0 + sinr)  # [users, subcarriers]
        rates = np.sum(rates, axis=1) / 1e3
        return rates

    def _build_observation(self, time_index: int) -> np.ndarray:
        """构建简化的观察空间。

        每个 agent 的观察包含：
        1. 所有 AP 的位置 (NUM_APS × 2)
        2. 所有 AP 的下倾角 (NUM_APS)
        3. st 时刻所有用户位置 (NUM_USERS × 2)
        4. st+1 时刻所有用户位置 (NUM_USERS × 2)
        5. 所有用户速率 (NUM_USERS)
        """
        assert self._scenario is not None
        prev_time_index = max(time_index - 1, 0)
        obs = np.zeros((NUM_APS, OBS_VECTOR_SIZE), dtype=np.float32)
        offset = 0

        # 所有 AP 的坐标 (展开为 NUM_APS × 2)
        ap_positions_flat = self._scenario.ap_metric_positions.reshape(-1).astype(np.float32)
        obs[:, offset : offset + NUM_APS * 2] = ap_positions_flat
        offset += NUM_APS * 2

        # 所有 AP 的下倾角（度）
        ap_downtilt_deg = np.degrees(self._scenario.ap_downtilt).astype(np.float32)
        obs[:, offset : offset + NUM_APS] = ap_downtilt_deg
        offset += NUM_APS

        # st 时刻所有用户位置
        user_pos_prev = self._scenario.user_positions[prev_time_index].reshape(-1).astype(np.float32)
        obs[:, offset : offset + NUM_USERS * 2] = user_pos_prev
        offset += NUM_USERS * 2

        # st+1 时刻所有用户位置
        user_pos_curr = self._scenario.user_positions[time_index].reshape(-1).astype(np.float32)
        obs[:, offset : offset + NUM_USERS * 2] = user_pos_curr
        offset += NUM_USERS * 2

        # 所有用户速率（全局信息）
        obs[:, offset : offset + NUM_USERS] = self.last_user_rates.astype(np.float32)
        offset += NUM_USERS

        return obs

    def _update_subcarrier_rates(self, user_rates: np.ndarray):
        assert self._scenario is not None
        users = self._scenario.subcarrier_users
        rates = np.zeros_like(users, dtype=np.float32)
        valid = (users >= 0) & (users < len(user_rates))
        rates[valid] = user_rates[users[valid]]
        self.last_ap_subcarrier_rates = rates

    def _log_ap_user_allocations(self, time_step: int):
        """打印每个 AP 的子载波用户分配情况及不同用户数量。"""
        assert self._scenario is not None
        for m in range(NUM_APS):
            users = self._scenario.subcarrier_users[m]
            unique_users = np.unique(users)
            print(
                f"step {time_step} - AP {m}:  "
                f"unique_user_count={len(unique_users)}"
            )

    def _log_user_rates(self, time_step: int, user_rates: np.ndarray):
        """按速率从高到低打印所有用户的速率。"""
        sorted_idx = np.argsort(-user_rates)
        sorted_rates = [(int(uid), float(user_rates[uid])) for uid in sorted_idx]
        print(f"step {time_step} - user rates (kbps) desc: {sorted_rates}")

    @staticmethod
    def _wrap_angle_deg(angle: np.ndarray) -> np.ndarray:
        return (angle + 180.0) % 360.0 - 180.0

    def _compute_gains_for_step(self, time_index: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._scenario is not None
        user_xy = self._scenario.user_metric_positions[time_index]  # (users, 2)
        ap_xy = self._scenario.ap_metric_positions  # (aps, 2)

        diff = user_xy[:, None, :] - ap_xy[None, :, :]
        horizontal_distance = np.linalg.norm(diff, axis=2)
        horizontal_distance = np.maximum(horizontal_distance, 0.1)

        az_to_user = np.degrees(np.arctan2(diff[:, :, 0], diff[:, :, 1]))
        azimuth_deg = np.degrees(self._scenario.ap_azimuth)[None, :]
        az_diff = self._wrap_angle_deg(az_to_user - azimuth_deg)

        elevation = np.degrees(np.arctan2(AP_HEIGHT_M - USER_HEIGHT_M, horizontal_distance))
        downtilt_deg = np.degrees(self._scenario.ap_downtilt)[None, :]
        theta_diff = elevation - downtilt_deg

        a_v = -np.minimum(12.0 * (theta_diff / THETA_3DB_DEG) ** 2, SIDELOBE_ATTEN_DB)
        a_h = -np.minimum(12.0 * (az_diff / PHI_3DB_DEG) ** 2, MAX_ATTEN_DB)
        attenuation = -np.minimum(-(a_v + a_h), MAX_ATTEN_DB)

        gains_dbi = GT_MAX_DBI + attenuation
        gains_linear = 10 ** (gains_dbi / 10.0)
        return gains_dbi, gains_linear

    def _build_power_tensor(self) -> np.ndarray:
        assert self._scenario is not None
        tensor = np.zeros((NUM_USERS, NUM_APS, NUM_SUBCARRIERS), dtype=float)
        su = self._scenario.subcarrier_users
        powers = self._scenario.subcarrier_powers_w
        valid = (su >= 0) & (su < NUM_USERS)
        ap_idx, sub_idx = np.nonzero(valid)
        if len(ap_idx) > 0:
            user_idx = su[ap_idx, sub_idx]
            tensor[user_idx, ap_idx, sub_idx] = powers[ap_idx, sub_idx]
        return tensor


class MultiAgentCellFreeEnv(gym.Env):
    """Multi-agent wrapper to align CellFreeEnv with MAPPO interfaces."""

    metadata = {"render_modes": []}

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self._env = CellFreeEnv()
        self.num_agents = NUM_APS
        self.num_users = NUM_USERS
        action_dim = self._env.action_space.shape[1]
        obs_dim = self._env.observation_space.shape[1]

        box_low = self._env.action_space.low[0]
        box_high = self._env.action_space.high[0]
        self.action_space = [
            spaces.Box(low=box_low, high=box_high, shape=(action_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        obs_low = np.full((obs_dim,), -np.inf, dtype=np.float32)
        obs_high = np.full((obs_dim,), np.inf, dtype=np.float32)
        self.observation_space = [
            spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        share_dim = obs_dim * self.num_agents
        share_low = np.full((share_dim,), -np.inf, dtype=np.float32)
        share_high = np.full((share_dim,), np.inf, dtype=np.float32)
        self.share_observation_space = [
            spaces.Box(low=share_low, high=share_high, dtype=np.float32)
        ]

        if seed is not None:
            self.seed(seed)

    def seed(self, seed: Optional[int] = None):
        self._env._rng = np.random.default_rng(seed if seed is not None else RANDOM_SEED)

    def reset(self):
        obs, _ = self._env.reset()
        return np.asarray(obs, dtype=np.float32)

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        obs, rewards, terminated, truncated, info = self._env.step(actions)
        obs = np.asarray(obs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        if rewards.ndim == 1:
            rewards = rewards.reshape(self.num_agents, -1)
        dones = np.full((self.num_agents,), bool(terminated or truncated))
        infos = [dict(info) if isinstance(info, dict) else {} for _ in range(self.num_agents)]
        return obs, rewards, dones, infos

    def close(self):
        self._env.close()
