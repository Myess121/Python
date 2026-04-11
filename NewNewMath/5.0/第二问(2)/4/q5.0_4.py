# -*- coding: utf-8 -*-
"""
参数扫描 + 断点续跑 + 岛屿陆地掩码
用于三台电脑分别运行不同参数组合
"""

import numpy as np
import random
import time
import json
import os
import sys
from multiprocessing import Pool, cpu_count
from numba import jit
from pathlib import Path

# ======================== 固定参数 ========================
M, N = 16, 23                     # 网格数 (纬度, 经度)
dx, dy = 20.0, 20.0               # 网格边长 (km)
V_uav = 180.0                     # km/h
dt = dx / V_uav                    # 时间步长 (h)
max_flight_time = 40.0             # 最大续航 (h)
max_steps = int(max_flight_time / dt)   # 360步
start_step = 23                    # 第23步开始探测（前22步入场）

Pd = 0.95                          # 探测概率
Pf = 0.05                          # 虚警概率

# 目标运动参数
p_stay = 0.95
p_move = 0.05

H = 6                              # 规划步数（固定）
w_start = 0.9
c1 = c2 = 1.5

# ======================== 岛屿陆地掩码 (根据经纬度) ========================
# 区域经纬度范围 (左上角 25°N,124°E ; 右下角 21°N,127°E)
lat_top = 25.0
lat_bottom = 21.0
lon_left = 124.0
lon_right = 127.0

# 岛屿矩形 [lat_min, lat_max, lon_min, lon_max] (纬度从小到大，经度从小到大)
islands_deg = [
    [24.3, 24.5, 124.1, 124.2],   # 岛屿1
    [24.7, 24.8, 125.2, 125.4],   # 岛屿2
    [24.63, 24.67, 124.6, 124.7]  # 岛屿3
]

def latlon_to_grid(lat, lon):
    """经纬度转网格索引 (1-based)"""
    i = int((lat_top - lat) / (lat_top - lat_bottom) * (M - 1)) + 1
    j = int((lon - lon_left) / (lon_right - lon_left) * (N - 1)) + 1
    return max(1, min(i, M)), max(1, min(j, N))

# 构建陆地掩码 (M x N, True=陆地)
land_mask = np.zeros((M, N), dtype=bool)
for (lat_min, lat_max, lon_min, lon_max) in islands_deg:
    i_min, j_min = latlon_to_grid(lat_max, lon_min)   # 纬度大的对应行号小
    i_max, j_max = latlon_to_grid(lat_min, lon_max)
    for i in range(min(i_min, i_max), max(i_min, i_max) + 1):
        for j in range(min(j_min, j_max), max(j_min, j_max) + 1):
            land_mask[i-1, j-1] = True

# ======================== 辅助函数（含陆地约束） ========================
def get_neighbors(i, j, include_land=False):
    """返回四邻域网格列表，默认排除陆地（如果 include_land=True 则包括）"""
    neighbors = []
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni, nj = i+di, j+dj
        if 1 <= ni <= M and 1 <= nj <= N:
            if include_land or not land_mask[ni-1, nj-1]:
                neighbors.append((ni, nj))
    return neighbors

def build_transition_matrix():
    """构建转移概率矩阵，禁止转移到陆地"""
    n_cells = M * N
    T = np.zeros((n_cells, n_cells))
    for idx in range(n_cells):
        i = idx // N + 1
        j = idx % N + 1
        # 如果当前网格是陆地，则目标不可能在这里（概率始终0），但矩阵中这一列全零即可
        if land_mask[i-1, j-1]:
            continue
        row_idx = (i-1)*N + (j-1)
        T[row_idx, idx] = p_stay
        neighbors = get_neighbors(i, j, include_land=False)
        if neighbors:
            prob = p_move / len(neighbors)
            for (ni, nj) in neighbors:
                row_idx2 = (ni-1)*N + (nj-1)
                T[row_idx2, idx] = prob
    # 归一化：对于非陆地列，确保和为1；对于陆地列，全零（因为目标不会在陆地）
    for idx in range(n_cells):
        col_sum = T[:, idx].sum()
        if col_sum > 0 and col_sum != 1.0:
            T[:, idx] /= col_sum
    return T

T_mat = build_transition_matrix()

@jit(nopython=True)
def predict_probability_numba(p_flat, T_mat):
    return T_mat @ p_flat

def apply_land_mask(p):
    """将陆地网格概率置零并重新归一化"""
    p[land_mask] = 0.0
    total = p.sum()
    if total > 0:
        p /= total
    return p

def bayesian_update(p, detected, i, j):
    """贝叶斯更新（原地修改）"""
    # 如果探测网格是陆地，不更新（目标不可能在那里）
    if land_mask[i-1, j-1]:
        return p
    prior = p[i-1, j-1]
    if detected:
        post = (prior * Pd) / (prior * Pd + (1-prior) * Pf)
    else:
        post = (prior * (1-Pd)) / (prior * (1-Pd) + (1-prior) * (1-Pf))
    p[i-1, j-1] = post
    # 归一化（保持总和为1）
    p /= p.sum()
    # 再次确保陆地概率为0
    apply_land_mask(p)
    return p

def generate_observation(true_pos, probe_pos):
    """生成观测值"""
    if true_pos == probe_pos:
        return random.random() < Pd
    else:
        return random.random() < Pf

def move_target(i, j):
    """目标移动，禁止进入陆地"""
    if random.random() < p_stay:
        return i, j
    neighbors = get_neighbors(i, j, include_land=False)
    if neighbors:
        return random.choice(neighbors)
    return i, j

@jit(nopython=True)
def decode_dirs_numba(dirs, start_i, start_j, M, N):
    traj_i = np.zeros(len(dirs)+1, dtype=np.int32)
    traj_j = np.zeros(len(dirs)+1, dtype=np.int32)
    traj_i[0] = start_i
    traj_j[0] = start_j
    ci, cj = start_i, start_j
    for idx, d in enumerate(dirs):
        if d == 0: ci = max(1, ci-1)
        elif d == 1: ci = min(M, ci+1)
        elif d == 2: cj = max(1, cj-1)
        elif d == 3: cj = min(N, cj+1)
        traj_i[idx+1] = ci
        traj_j[idx+1] = cj
    return traj_i, traj_j

# ======================== CC-MPSO 类 ========================
class CC_MPSO_Fast:
    def __init__(self, K, H, pop_size, max_iter, coop_gap, w_start, w_end, c1, c2):
        self.K = K
        self.H = H
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.coop_gap = coop_gap
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.subpop = []
        for _ in range(K):
            particles = np.random.randint(0, 4, size=(pop_size, H))
            velocities = np.random.uniform(-1, 1, size=(pop_size, H))
            pbest = particles.copy()
            pbest_fit = np.full(pop_size, -np.inf)
            gbest = particles[0].copy()
            gbest_fit = -np.inf
            self.subpop.append({
                'particles': particles, 'velocities': velocities,
                'pbest': pbest, 'pbest_fit': pbest_fit,
                'gbest': gbest, 'gbest_fit': gbest_fit
            })

    def evaluate_fast(self, all_dirs, start_positions, p_curr, T_mat, M, N, Pd, penalty_coef, land_mask):
        p_seq = [p_curr.flatten()]
        for _ in range(self.H):
            p_seq.append(predict_probability_numba(p_seq[-1], T_mat))
        total_reward = 0.0
        penalty = 0.0
        for h in range(self.H):
            positions = []
            for k in range(self.K):
                traj_i, traj_j = decode_dirs_numba(all_dirs[k],
                                                   start_positions[k][0],
                                                   start_positions[k][1],
                                                   M, N)
                pos = (traj_i[h+1], traj_j[h+1])
                # 如果无人机飞到了陆地，给予惩罚（避免选陆地）
                if land_mask[pos[0]-1, pos[1]-1]:
                    penalty += penalty_coef
                positions.append(pos)
            unique_pos = set(positions)
            if len(unique_pos) != len(positions):
                penalty += penalty_coef
            p_flat = p_seq[h+1]
            for k, (i, j) in enumerate(positions):
                idx = (i-1)*N + (j-1)
                total_reward += p_flat[idx]
        return total_reward * Pd - penalty

    def optimize(self, start_positions, p_curr, T_mat, M, N, Pd, penalty_coef, land_mask):
        # 初始化评估
        for sidx, sub in enumerate(self.subpop):
            for i in range(self.pop_size):
                dirs = sub['particles'][i]
                all_dirs = []
                for k in range(self.K):
                    if k == sidx:
                        all_dirs.append(dirs)
                    else:
                        all_dirs.append(self.subpop[k]['gbest'])
                fit = self.evaluate_fast(all_dirs, start_positions, p_curr,
                                         T_mat, M, N, Pd, penalty_coef, land_mask)
                if fit > sub['pbest_fit'][i]:
                    sub['pbest_fit'][i] = fit
                    sub['pbest'][i] = dirs.copy()
                if fit > sub['gbest_fit']:
                    sub['gbest_fit'] = fit
                    sub['gbest'] = dirs.copy()
        # 迭代优化
        for it in range(self.max_iter):
            w = self.w_start - (self.w_start - self.w_end) * it / self.max_iter
            for sidx, sub in enumerate(self.subpop):
                for i in range(self.pop_size):
                    r1, r2 = random.random(), random.random()
                    vel = (w * sub['velocities'][i] +
                           self.c1 * r1 * (sub['pbest'][i] - sub['particles'][i]) +
                           self.c2 * r2 * (sub['gbest'] - sub['particles'][i]))
                    sub['velocities'][i] = vel
                    new_p = sub['particles'][i] + vel
                    new_p = np.clip(new_p, 0, 3)
                    new_p = np.round(new_p).astype(np.int32)
                    sub['particles'][i] = new_p
                for i in range(self.pop_size):
                    dirs = sub['particles'][i]
                    all_dirs = []
                    for k in range(self.K):
                        if k == sidx:
                            all_dirs.append(dirs)
                        else:
                            all_dirs.append(self.subpop[k]['gbest'])
                    fit = self.evaluate_fast(all_dirs, start_positions, p_curr,
                                             T_mat, M, N, Pd, penalty_coef, land_mask)
                    if fit > sub['pbest_fit'][i]:
                        sub['pbest_fit'][i] = fit
                        sub['pbest'][i] = dirs.copy()
                    if fit > sub['gbest_fit']:
                        sub['gbest_fit'] = fit
                        sub['gbest'] = dirs.copy()
        return [sub['gbest'] for sub in self.subpop]

# ======================== 单次仿真（含陆地约束） ========================
def run_single_simulation(K, target_init_pos, uav_start_positions, params):
    pop_size = params['pop_size']
    max_iter = params['max_iter']
    penalty_coef = params['penalty_coef']
    coop_gap = params['coop_gap']
    w_end = params['w_end']

    p = np.ones((M, N)) / (M * N)
    apply_land_mask(p)               # 陆地概率置零

    target_i, target_j = target_init_pos
    uav_pos_array = np.array(uav_start_positions, dtype=np.int32)
    total_steps_done = start_step - 1
    detected = False
    discovery_step = None

    while total_steps_done < max_steps and not detected:
        remaining = max_steps - total_steps_done
        horizon = min(H, remaining)
        if horizon <= 0:
            break
        optimizer = CC_MPSO_Fast(K, horizon, pop_size, max_iter, coop_gap,
                                 w_start, w_end, c1, c2)
        start_positions = [(uav_pos_array[k][0], uav_pos_array[k][1]) for k in range(K)]
        best_dirs = optimizer.optimize(start_positions, p, T_mat, M, N, Pd, penalty_coef, land_mask)

        for step in range(horizon):
            if total_steps_done >= max_steps or detected:
                break
            new_positions = []
            for k in range(K):
                d = best_dirs[k][step]
                ci, cj = uav_pos_array[k]
                if d == 0: ci = max(1, ci-1)
                elif d == 1: ci = min(M, ci+1)
                elif d == 2: cj = max(1, cj-1)
                elif d == 3: cj = min(N, cj+1)
                new_positions.append((ci, cj))

            for k, (ci, cj) in enumerate(new_positions):
                obs = generate_observation((target_i, target_j), (ci, cj))
                p = bayesian_update(p, obs, ci, cj)
                if obs and ci == target_i and cj == target_j:
                    detected = True
                    discovery_step = total_steps_done + 1
                    break
            if detected:
                break

            uav_pos_array = np.array(new_positions, dtype=np.int32)
            total_steps_done += 1
            target_i, target_j = move_target(target_i, target_j)
            # 概率传播
            p_flat = p.flatten()
            p_flat = predict_probability_numba(p_flat, T_mat)
            p = p_flat.reshape((M, N))
            apply_land_mask(p)      # 确保陆地概率为0
    return discovery_step if detected else max_steps

# ======================== 断点续跑函数 ========================
def run_config(params, target_sims, result_file, K=2):
    steps_list = []
    start_idx = 0
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                steps_list = data.get('steps_list', [])
                start_idx = len(steps_list)
                print(f"发现已有结果文件，已完成 {start_idx} 次，继续...")
        except:
            print("读取已有文件失败，从头开始")
            start_idx = 0
            steps_list = []

    random.seed(12345 + hash(frozenset(params.items())) % 10000)
    np.random.seed(12345)

    uav_start = [(1,1)] * K   # 所有无人机从左上角进入（可改进，但保持原样）

    try:
        for sim_idx in range(start_idx, target_sims):
            # 随机生成非陆地的目标初始位置
            while True:
                i = random.randint(1, M)
                j = random.randint(1, N)
                if not land_mask[i-1, j-1]:
                    break
            init_pos = (i, j)
            step = run_single_simulation(K, init_pos, uav_start, params)
            steps_list.append(step)

            if (sim_idx + 1) % 10 == 0:
                with open(result_file, 'w') as f:
                    json.dump({'params': params, 'steps_list': steps_list, 'completed': sim_idx+1}, f, indent=2)
                print(f"  已保存进度: {sim_idx+1}/{target_sims}")
        # 最终保存
        with open(result_file, 'w') as f:
            json.dump({'params': params, 'steps_list': steps_list, 'completed': target_sims}, f, indent=2)
        print(f"配置完成: {params}")
    except KeyboardInterrupt:
        print(f"\n用户中断，已保存进度到 {result_file}")
        sys.exit(0)

    steps_arr = np.array(steps_list)
    avg_step = np.mean(steps_arr)
    std_step = np.std(steps_arr)
    avg_time = avg_step * dt
    std_time = std_step * dt
    rate_10h = np.sum(steps_arr < (10.0/dt)) / target_sims * 100
    rate_40h = np.sum(steps_arr < max_steps) / target_sims * 100

    summary = {
        'params': params,
        'target_sims': target_sims,
        'avg_time_hours': avg_time,
        'std_time_hours': std_time,
        'rate_within_10h': rate_10h,
        'rate_within_40h': rate_40h,
    }
    summary_file = result_file.replace('.json', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n结果摘要: {summary}")
    return summary

# ======================== 主程序 ========================
# 定义所有参数组合（12组）
PARAM_SWEEP = [
    {'pop_size': 15, 'max_iter': 40, 'penalty_coef': 1000, 'coop_gap': 5, 'w_end': 0.4},
    {'pop_size': 15, 'max_iter': 80, 'penalty_coef': 1000, 'coop_gap': 5, 'w_end': 0.4},
    {'pop_size': 15, 'max_iter': 120, 'penalty_coef': 1000, 'coop_gap': 5, 'w_end': 0.4},
    {'pop_size': 30, 'max_iter': 40, 'penalty_coef': 1000, 'coop_gap': 5, 'w_end': 0.4},
    {'pop_size': 30, 'max_iter': 80, 'penalty_coef': 1000, 'coop_gap': 5, 'w_end': 0.4},
    {'pop_size': 30, 'max_iter': 120, 'penalty_coef': 1000, 'coop_gap': 5, 'w_end': 0.4},
    {'pop_size': 50, 'max_iter': 40, 'penalty_coef': 1000, 'coop_gap': 5, 'w_end': 0.4},
    {'pop_size': 50, 'max_iter': 80, 'penalty_coef': 1000, 'coop_gap': 5, 'w_end': 0.4},
    {'pop_size': 50, 'max_iter': 120, 'penalty_coef': 1000, 'coop_gap': 5, 'w_end': 0.4},
    {'pop_size': 30, 'max_iter': 80, 'penalty_coef': 500, 'coop_gap': 5, 'w_end': 0.4},
    {'pop_size': 30, 'max_iter': 80, 'penalty_coef': 2000, 'coop_gap': 5, 'w_end': 0.4},
    {'pop_size': 30, 'max_iter': 80, 'penalty_coef': 1000, 'coop_gap': 10, 'w_end': 0.4},
]

TARGET_SIMS = 50   # 每组目标仿真次数

if __name__ == "__main__":
    print("预热 Numba...")
    _ = predict_probability_numba(np.ones(M*N), T_mat)
    print("预热完成。")

    print("\n可用参数组合索引 (0~11):")
    for idx, p in enumerate(PARAM_SWEEP):
        print(f"{idx}: {p}")
    sel = input("请输入要运行的组合索引，多个用逗号分隔 (例如 0,1,2): ")
    indices = [int(x.strip()) for x in sel.split(',')]

    for idx in indices:
        params = PARAM_SWEEP[idx]
        fname = f"results_K2_pop{params['pop_size']}_iter{params['max_iter']}_pen{params['penalty_coef']}_coop{params['coop_gap']}_wend{params['w_end']}.json"
        print(f"\n开始运行组合 {idx}: {params}")
        run_config(params, TARGET_SIMS, fname, K=4)
        print(f"组合 {idx} 完成！\n")

    print("所有指定组合运行完毕！")
    # 0,1,2,3,4,5,6,7,8,9,10,11