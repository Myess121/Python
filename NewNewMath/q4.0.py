# -*- coding: utf-8 -*-
"""
参数扫描 + 断点续跑 + CPU (Numba加速)
支持多无人机数量 K = 2 ~ 10，每组仿真次数 500
完全保留 CC-MPSO 算法，无 GPU 依赖
"""

import numpy as np
import random
import time
import json
import os
import sys
from pathlib import Path
from numba import jit

# ======================== 固定参数 ========================
M, N = 16, 23
dx, dy = 20.0, 20.0
V_uav = 180.0
dt = dx / V_uav
max_flight_time = 40.0
max_steps = int(max_flight_time / dt)
start_step = 23
Pd = 0.95
Pf = 0.05
p_stay = 0.95
p_move = 0.05
H = 6                     # 规划步数
w_start = 0.9
c1 = c2 = 1.5

# ======================== 辅助函数 ========================
def get_neighbors(i, j):
    neighbors = []
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni, nj = i+di, j+dj
        if 1 <= ni <= M and 1 <= nj <= N:
            neighbors.append((ni, nj))
    return neighbors

def build_transition_matrix(M, N, p_stay, p_move):
    n_cells = M * N
    T = np.zeros((n_cells, n_cells), dtype=np.float64)  # 改为 float64
    for idx in range(n_cells):
        i = idx // N + 1
        j = idx % N + 1
        row_idx = (i-1)*N + (j-1)
        T[row_idx, idx] = p_stay
        neighbors = get_neighbors(i, j)
        if neighbors:
            prob = p_move / len(neighbors)
            for (ni, nj) in neighbors:
                row_idx2 = (ni-1)*N + (nj-1)
                T[row_idx2, idx] = prob
    return T

T_mat = build_transition_matrix(M, N, p_stay, p_move)

@jit(nopython=True)
def predict_probability_numba(p_flat, T_mat):
    return T_mat @ p_flat

@jit(nopython=True)
def bayesian_update_numba(p_grid, detected, i, j, Pd, Pf):
    prior = p_grid[i-1, j-1]
    if detected:
        post = (prior * Pd) / (prior * Pd + (1-prior) * Pf)
    else:
        post = (prior * (1-Pd)) / (prior * (1-Pd) + (1-prior) * (1-Pf))
    p_grid[i-1, j-1] = post
    total = np.sum(p_grid)
    if total > 0:
        p_grid /= total
    return p_grid

@jit(nopython=True)
def move_target_numba(i, j, p_stay, p_move, M, N):
    if np.random.random() < p_stay:
        return i, j
    neighbors = []
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni, nj = i+di, j+dj
        if 1 <= ni <= M and 1 <= nj <= N:
            neighbors.append((ni, nj))
    if neighbors:
        idx = np.random.randint(0, len(neighbors))
        return neighbors[idx]
    return i, j

@jit(nopython=True)
def generate_observation_numba(true_i, true_j, probe_i, probe_j, Pd, Pf):
    if true_i == probe_i and true_j == probe_j:
        return np.random.random() < Pd
    else:
        return np.random.random() < Pf

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

    def evaluate_fast(self, all_dirs, start_positions, p_curr, T_mat, M, N, Pd, penalty_coef):
        # 确保 p_curr 是 float64
        p_curr_flat = p_curr.flatten().astype(np.float64)
        p_seq = [p_curr_flat]
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
                positions.append((traj_i[h+1], traj_j[h+1]))
            unique_pos = set(positions)
            if len(unique_pos) != len(positions):
                penalty += penalty_coef
            p_flat = p_seq[h+1]
            for k, (i, j) in enumerate(positions):
                idx = (i-1)*N + (j-1)
                total_reward += p_flat[idx]
        return total_reward * Pd - penalty

    def optimize(self, start_positions, p_curr, T_mat, M, N, Pd, penalty_coef):
        # 初始化
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
                                         T_mat, M, N, Pd, penalty_coef)
                if fit > sub['pbest_fit'][i]:
                    sub['pbest_fit'][i] = fit
                    sub['pbest'][i] = dirs.copy()
                if fit > sub['gbest_fit']:
                    sub['gbest_fit'] = fit
                    sub['gbest'] = dirs.copy()
        # 迭代
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
                                             T_mat, M, N, Pd, penalty_coef)
                    if fit > sub['pbest_fit'][i]:
                        sub['pbest_fit'][i] = fit
                        sub['pbest'][i] = dirs.copy()
                    if fit > sub['gbest_fit']:
                        sub['gbest_fit'] = fit
                        sub['gbest'] = dirs.copy()
        return [sub['gbest'] for sub in self.subpop]

# ======================== 单次仿真 ========================
def run_single_simulation(K, target_init_pos, uav_start_positions, params):
    pop_size = params['pop_size']
    max_iter = params['max_iter']
    penalty_coef = params['penalty_coef']
    coop_gap = params['coop_gap']
    w_end = params['w_end']

    p = np.ones((M, N), dtype=np.float64) / (M * N)   # float64
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
        best_dirs = optimizer.optimize(start_positions, p, T_mat, M, N, Pd, penalty_coef)
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
                obs = generate_observation_numba(target_i, target_j, ci, cj, Pd, Pf)
                p = bayesian_update_numba(p, obs, ci, cj, Pd, Pf)
                if obs and ci == target_i and cj == target_j:
                    detected = True
                    discovery_step = total_steps_done + 1
                    break
            if detected:
                break
            uav_pos_array = np.array(new_positions, dtype=np.int32)
            total_steps_done += 1
            target_i, target_j = move_target_numba(target_i, target_j, p_stay, p_move, M, N)
            p_flat = p.flatten()
            p_flat = predict_probability_numba(p_flat, T_mat)
            p = p_flat.reshape((M, N))
    return discovery_step if detected else max_steps

# ======================== 批量运行配置 ========================
def run_config(params, target_sims, result_file, K):
    steps_list = []
    start_idx = 0
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                steps_list = data.get('steps_list', [])
                start_idx = len(steps_list)
                print(f"发现已有结果文件，已完成 {start_idx} 次仿真，将继续从第 {start_idx+1} 次开始")
        except:
            print("读取已有结果文件失败，从头开始")
            start_idx = 0
            steps_list = []

    random.seed(12345)
    np.random.seed(12345)

    uav_start = [(1,1)] * K

    try:
        for sim_idx in range(start_idx, target_sims):
            init_pos = (random.randint(1, M), random.randint(1, N))
            step = run_single_simulation(K, init_pos, uav_start, params)
            steps_list.append(step)

            if (sim_idx + 1) % 10 == 0:
                with open(result_file, 'w') as f:
                    json.dump({'params': params, 'steps_list': steps_list, 'completed': sim_idx+1}, f, indent=2)
                print(f"  K={K} 已保存进度: {sim_idx+1}/{target_sims}")

        with open(result_file, 'w') as f:
            json.dump({'params': params, 'steps_list': steps_list, 'completed': target_sims}, f, indent=2)
        print(f"配置完成: K={K}, params={params}")
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
        'K': K,
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

    print(f"\n结果摘要 (K={K}): {summary}")
    return summary

# ======================== 主程序 ========================
# 参数组合
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

TARGET_SIMS = 50
K_LIST = list(range(2, 11))

if __name__ == "__main__":
    print("预热 Numba...")
    # 预热时需要传入正确 dtype 的数组
    dummy_flat = np.ones(M*N, dtype=np.float64)
    _ = predict_probability_numba(dummy_flat, T_mat)
    print("预热完成。")

    print(f"\n将依次运行 K = {K_LIST}，每组 {TARGET_SIMS} 次仿真，共 {len(K_LIST)} 组。")
    print("参数组合共 {} 组。".format(len(PARAM_SWEEP)))
    print("建议先输入单个 K 和单个参数索引进行测试。")

    sel_k = input(f"\n请输入要运行的 K 值，多个用逗号分隔 (默认全部 {K_LIST}): ")
    if sel_k.strip():
        k_list = [int(x.strip()) for x in sel_k.split(',')]
    else:
        k_list = K_LIST

    sel_param = input(f"请输入要运行的参数组合索引，多个用逗号分隔 (默认全部 0-{len(PARAM_SWEEP)-1}): ")
    if sel_param.strip():
        param_indices = [int(x.strip()) for x in sel_param.split(',')]
    else:
        param_indices = list(range(len(PARAM_SWEEP)))

    for K in k_list:
        print(f"\n{'='*60}")
        print(f"开始处理 K = {K}")
        print(f"{'='*60}")
        for idx in param_indices:
            params = PARAM_SWEEP[idx]
            fname = f"results_K{K}_pop{params['pop_size']}_iter{params['max_iter']}_pen{params['penalty_coef']}_coop{params['coop_gap']}_wend{params['w_end']}.json"
            print(f"\n运行组合 K={K}, param_idx={idx}: {params}")
            run_config(params, TARGET_SIMS, fname, K)
        print(f"K={K} 的所有参数组合运行完毕！")

    print("\n所有指定 K 和参数组合运行完毕！")