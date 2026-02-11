from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import (
    AP_HEIGHT_M,
    AP_POWER_BUDGET_W,
    AREA_SIZE_M,
    CARRIER_BANDWIDTH_HZ,
    CARRIER_FREQUENCY_HZ,
    DIRECTION_PERSIST_PROB,
    GRID_CELL_SIZE_M,
    GRID_POINT_SPACING_M,
    GT_MAX_DBI,
    HOPS_PER_STEP,
    MAX_ATTEN_DB,
    NOISE_POWER_W,
    NUM_APS,
    NUM_SUBCARRIERS,
    NUM_USERS,
    PHI_3DB_DEG,
    SHADOW_FADING_STD_DB,
    SIDELOBE_ATTEN_DB,
    THETA_3DB_DEG,
    TIME_STEPS,
    USER_HEIGHT_M,
)

_GRID_VALUES = np.arange(0.0, AREA_SIZE_M + GRID_POINT_SPACING_M, GRID_POINT_SPACING_M)
_GRID_MAX_IDX = len(_GRID_VALUES) - 1
_NEIGHBOR_STEPS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _valid_neighbors(x_idx: int, y_idx: int) -> List[Tuple[int, int]]:
    neighbors: List[Tuple[int, int]] = []
    for dx, dy in _NEIGHBOR_STEPS:
        nx_idx = x_idx + dx
        ny_idx = y_idx + dy
        if 0 <= nx_idx <= _GRID_MAX_IDX and 0 <= ny_idx <= _GRID_MAX_IDX:
            neighbors.append((nx_idx, ny_idx))
    return neighbors


def generate_user_paths(
    num_users: int,
    rng: np.random.Generator,
    time_steps: int = TIME_STEPS,
) -> np.ndarray:
    """生成按 1 米网格移动的用户轨迹，每个时间步执行固定跳数。"""
    position_idx = rng.integers(0, len(_GRID_VALUES), size=(num_users, 2))
    visited = np.zeros((num_users, len(_GRID_VALUES), len(_GRID_VALUES)), dtype=bool)
    for u in range(num_users):
        visited[u, position_idx[u, 0], position_idx[u, 1]] = True

    directions = np.zeros((num_users, 2), dtype=int)
    positions = np.zeros((time_steps, num_users, 2), dtype=float)
    positions[0, :, 0] = _GRID_VALUES[position_idx[:, 0]]
    positions[0, :, 1] = _GRID_VALUES[position_idx[:, 1]]

    for t in range(1, time_steps):
        for u in range(num_users):
            cx, cy = int(position_idx[u, 0]), int(position_idx[u, 1])
            for _ in range(HOPS_PER_STEP):
                neighbors = _valid_neighbors(cx, cy)
                if not neighbors:
                    break

                unvisited = [p for p in neighbors if not visited[u, p[0], p[1]]]
                candidates = unvisited if unvisited else neighbors

                preferred = None
                dirx, diry = directions[u]
                if dirx != 0 or diry != 0:
                    forward = (cx + dirx, cy + diry)
                    if forward in candidates and rng.random() < DIRECTION_PERSIST_PROB:
                        preferred = forward

                if preferred is None:
                    choice = int(rng.integers(len(candidates)))
                    next_x, next_y = candidates[choice]
                else:
                    next_x, next_y = preferred

                directions[u] = (next_x - cx, next_y - cy)
                cx, cy = next_x, next_y
                visited[u, cx, cy] = True

            position_idx[u] = (cx, cy)
            positions[t, u, 0] = _GRID_VALUES[cx]
            positions[t, u, 1] = _GRID_VALUES[cy]

    return positions


def place_access_points(rng: np.random.Generator) -> Tuple[List[Dict], np.ndarray]:
    """在 10x10 网格中以远离已选点的方式放置 AP，尽量均匀分布。"""
    cells_per_axis = int(AREA_SIZE_M / GRID_CELL_SIZE_M)
    if cells_per_axis * GRID_CELL_SIZE_M != AREA_SIZE_M:
        raise ValueError("AREA_SIZE_M 必须能被 GRID_CELL_SIZE_M 整除。")

    grid_indices = [(i, j) for i in range(cells_per_axis) for j in range(cells_per_axis)]
    centers = np.array(
        [
            ((gx + 0.5) * GRID_CELL_SIZE_M, (gy + 0.5) * GRID_CELL_SIZE_M)
            for gx, gy in grid_indices
        ],
        dtype=float,
    )

    first_idx = int(rng.integers(len(grid_indices)))
    selected_idx = [first_idx]
    while len(selected_idx) < NUM_APS:
        mask = np.ones(len(grid_indices), dtype=bool)
        mask[selected_idx] = False
        remaining = np.nonzero(mask)[0]

        # 计算每个候选点到已选点的最小距离，选取最远者以均匀覆盖
        min_dists = []
        for ridx in remaining:
            dists = np.linalg.norm(centers[ridx] - centers[selected_idx], axis=1)
            min_dists.append(np.min(dists))
        min_dists = np.array(min_dists)
        max_dist = np.max(min_dists)
        best = remaining[min_dists == max_dist]
        choice = int(rng.choice(best))
        selected_idx.append(choice)

    aps: List[Dict] = []
    ap_positions = np.zeros((NUM_APS, 2), dtype=float)
    equal_power_w = AP_POWER_BUDGET_W / NUM_SUBCARRIERS
    for ap_idx, sel in enumerate(selected_idx):
        gx, gy = grid_indices[int(sel)]
        x, y = centers[int(sel)]
        azimuth = 0.0  # 默认指向正北
        downtilt = 0.0
        subcarrier_users = [(n % NUM_USERS) for n in range(NUM_SUBCARRIERS)]
        powers_w = [equal_power_w for _ in range(NUM_SUBCARRIERS)]
        powers_dbm = [10 * math.log10(pw) + 30.0 for pw in powers_w]
        aps.append(
            {
                "source_id": f"ap_{ap_idx}",
                "grid_idx": (gx, gy),
                "x": x,
                "y": y,
                "azimuth": azimuth,
                "downtilt": downtilt,
                "subcarrier_users": subcarrier_users,
                "subcarrier_powers_dbm": powers_dbm,
                "subcarrier_powers_w": powers_w,
            }
        )
        ap_positions[ap_idx] = (x, y)
    return aps, ap_positions


def compute_channel_tensor(
    user_metric_positions: np.ndarray,
    ap_metric_positions: np.ndarray,
    rng: np.random.Generator,
    antenna_gains_linear: np.ndarray | None,
) -> np.ndarray:
    """计算所有时间/用户/AP/子载波的复信道系数。"""
    steps, num_users, _ = user_metric_positions.shape
    num_aps = ap_metric_positions.shape[0]
    channels = np.zeros((steps, num_users, num_aps, NUM_SUBCARRIERS), dtype=np.complex128)

    for t in range(steps):
        for k in range(num_users):
            ux, uy = user_metric_positions[t, k]
            for m in range(num_aps):
                ax, ay = ap_metric_positions[m]
                d2d = max(math.hypot(ax - ux, ay - uy), 10.0)
                pl_db = 17.3 + 24.9 * math.log10(CARRIER_FREQUENCY_HZ / 1e9) + 38.3 * math.log10(d2d)
                pl_linear = 10 ** (-pl_db / 10.0)
                shadow_factor = 10 ** (SHADOW_FADING_STD_DB * rng.normal() / 10.0)
                beta = math.sqrt(pl_linear * shadow_factor)
                g_real = rng.normal(scale=math.sqrt(0.5), size=NUM_SUBCARRIERS)
                g_imag = rng.normal(scale=math.sqrt(0.5), size=NUM_SUBCARRIERS)
                channels[t, k, m] = beta * (g_real + 1j * g_imag)
    return channels


def compute_user_rates(
    channels: np.ndarray,
    power_tensor: np.ndarray,
    gains_linear: np.ndarray,
) -> np.ndarray:
    """按原速率公式计算每个时间步的用户速率。"""
    steps, num_users, num_aps, num_subcarriers = channels.shape
    bandwidth_per_sub = CARRIER_BANDWIDTH_HZ / NUM_SUBCARRIERS
    noise = NOISE_POWER_W
    rates = np.zeros((steps, num_users), dtype=float)
    h_abs2 = np.abs(channels) ** 2

    for t in range(steps):
        for k in range(num_users):
            rate = 0.0
            for n in range(num_subcarriers):
                useful = np.sum(
                    power_tensor[k, :, n] * gains_linear[t, k, :] * h_abs2[t, k, :, n]
                )
                interference = 0.0
                for i in range(num_users):
                    if i == k:
                        continue
                    interference += np.sum(
                        power_tensor[i, :, n]
                        * gains_linear[t, k, :]
                        * h_abs2[t, k, :, n]
                    )
                sinr = useful / (noise + interference + 1e-15)
                rate += bandwidth_per_sub * math.log2(1.0 + sinr)
            rates[t, k] = rate
    return rates / 1e3  # 转为 kbps


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def build_power_tensor(aps: Sequence[Dict]) -> np.ndarray:
    """构建瓦特单位的功率张量 p_kmn。"""
    tensor = np.zeros((NUM_USERS, len(aps), NUM_SUBCARRIERS), dtype=float)
    for m, ap in enumerate(aps):
        for n, user_id in enumerate(ap["subcarrier_users"]):
            if 0 <= user_id < NUM_USERS:
                tensor[user_id, m, n] = ap["subcarrier_powers_w"][n]
    return tensor


def compute_antenna_gains(
    user_positions: np.ndarray,
    aps: Sequence[Dict],
    user_metric_positions: np.ndarray | None = None,
    ap_metric_positions: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算每个时间/用户/AP 的天线增益。"""
    if ap_metric_positions is None:
        ap_metric_positions = np.array([[ap["x"], ap["y"]] for ap in aps], dtype=float)
    if user_metric_positions is None:
        user_metric_positions = user_positions

    steps, num_users, _ = user_metric_positions.shape
    num_aps = len(aps)
    gains_dbi = np.zeros((steps, num_users, num_aps), dtype=float)

    for t in range(steps):
        user_xy = user_metric_positions[t]
        for k in range(num_users):
            ux, uy = user_xy[k]
            for m in range(num_aps):
                ax, ay = ap_metric_positions[m]
                dx = ux - ax
                dy = uy - ay
                horizontal_distance = max(math.hypot(dx, dy), 0.1)
                az_to_user = math.degrees(math.atan2(dx, dy))
                az_diff = wrap_angle_deg(
                    az_to_user - math.degrees(aps[m]["azimuth"])
                )
                downtilt_deg = math.degrees(aps[m]["downtilt"])
                elevation = math.degrees(math.atan2(AP_HEIGHT_M - USER_HEIGHT_M, horizontal_distance))
                theta_diff = elevation - downtilt_deg

                a_v = -min(
                    12.0 * (theta_diff / THETA_3DB_DEG) ** 2,
                    SIDELOBE_ATTEN_DB,
                )
                a_h = -min(
                    12.0 * (az_diff / PHI_3DB_DEG) ** 2,
                    MAX_ATTEN_DB,
                )
                attenuation = -min(-(a_v + a_h), MAX_ATTEN_DB)
                gains_dbi[t, k, m] = GT_MAX_DBI + attenuation

    gains_linear = 10 ** (gains_dbi / 10.0)
    return gains_dbi, gains_linear
