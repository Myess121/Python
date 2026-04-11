# -*- coding: utf-8 -*-
"""
参数扫描 + 断点续跑 + GPU加速 (PyTorch)
支持多无人机数量 K = 2 ~ 10
每组仿真次数 500 次
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import random
import time
import json
import os
import sys
from pathlib import Path
import torch

# ======================== 检查 GPU ========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")

# ======================== 固定参数（不变） ========================
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
H = 6                     # 规划步数（固定）
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
    T = np.zeros((n_cells, n_cells), dtype=np.float32)
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

# 构建转移矩阵（CPU numpy）
T_mat_np = build_transition_matrix(M, N, p_stay, p_move)
# 转移到 GPU 并转为 torch 张量
T_mat_gpu = torch.from_numpy(T_mat_np).float().to(DEVICE)

# Numba 加速的函数（用于目标移动、观测生成等轻量操作，保留）
from numba import jit

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

# ======================== 解码函数（同时支持 NumPy 和 PyTorch） ========================
def decode_dirs_torch(dirs, start_i, start_j, M, N):
    """
    批量解码动作序列（GPU版本）
    dirs: shape (pop_size, H) 的动作整数张量
    start_i, start_j: 标量起始坐标 (1-based)
    返回: positions: shape (pop_size, H+1, 2) 的坐标张量 (i, j)，1-based
    """
    pop_size, H = dirs.shape
    device = dirs.device
    # 初始化位置数组
    positions = torch.zeros((pop_size, H+1, 2), dtype=torch.int32, device=device)
    positions[:, 0, 0] = start_i
    positions[:, 0, 1] = start_j
    ci = start_i
    cj = start_j
    # 逐 step 更新（循环次数 = H，通常 H=6，可接受）
    for step in range(H):
        d = dirs[:, step]  # (pop_size,)
        # 上
        mask_up = (d == 0)
        # 下
        mask_down = (d == 1)
        # 左
        mask_left = (d == 2)
        # 右
        mask_right = (d == 3)
        # 更新 ci, cj
        ci = ci * (~(mask_up | mask_down)).int() + (ci-1)*mask_up.int() + (ci+1)*mask_down.int()
        ci = torch.clamp(ci, 1, M)
        cj = cj * (~(mask_left | mask_right)).int() + (cj-1)*mask_left.int() + (cj+1)*mask_right.int()
        cj = torch.clamp(cj, 1, N)
        positions[:, step+1, 0] = ci
        positions[:, step+1, 1] = cj
    return positions

# ======================== CC-MPSO 类（GPU加速版，支持任意 K） ========================
class CC_MPSO_GPU:
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
                'particles': particles,
                'velocities': velocities,
                'pbest': pbest,
                'pbest_fit': pbest_fit,
                'gbest': gbest,
                'gbest_fit': gbest_fit
            })
        # 将子群的粒子数据转移到 GPU（作为 torch 张量）
        self.to_gpu()

    def to_gpu(self):
        """将子群数据转换为 GPU 张量"""
        for sub in self.subpop:
            sub['particles_gpu'] = torch.from_numpy(sub['particles']).int().to(DEVICE)
            sub['velocities_gpu'] = torch.from_numpy(sub['velocities']).float().to(DEVICE)
            sub['pbest_gpu'] = torch.from_numpy(sub['pbest']).int().to(DEVICE)
            sub['pbest_fit_gpu'] = torch.from_numpy(sub['pbest_fit']).float().to(DEVICE)
            sub['gbest_gpu'] = torch.from_numpy(sub['gbest']).int().to(DEVICE)
            # gbest_fit 是标量，保留在 CPU
            # 同时保留 CPU 副本以便更新

    def evaluate_batch_gpu(self, sub_idx, start_positions, p_curr_gpu, penalty_coef):
        """
        批量评估子群 sub_idx 中所有粒子的适应度（GPU 并行）
        start_positions: list of (i,j) 起始位置 (1-based)，长度为 K
        p_curr_gpu: 当前概率分布 (M, N) 的 GPU 张量
        penalty_coef: 碰撞惩罚系数
        返回: fitness (pop_size,) 的 CPU numpy 数组
        """
        sub = self.subpop[sub_idx]
        particles = sub['particles_gpu']  # (pop_size, H)
        pop_size = self.pop_size
        H = self.H
        K = self.K

        # 1. 预测未来 H 步的概率分布（批量预测）
        p_seq = [p_curr_gpu.flatten()]
        for _ in range(H):
            p_next = torch.mv(T_mat_gpu, p_seq[-1])
            p_seq.append(p_next)
        p_seq_stack = torch.stack(p_seq, dim=0)  # (H+1, M*N)

        # 2. 为每个粒子构造完整的动作组合
        all_dirs = torch.zeros((pop_size, K, H), dtype=torch.int32, device=DEVICE)
        for k in range(K):
            if k == sub_idx:
                all_dirs[:, k, :] = particles
            else:
                gbest = self.subpop[k]['gbest_gpu']  # (H,)
                all_dirs[:, k, :] = gbest.unsqueeze(0).expand(pop_size, -1)

        # 3. 解码所有无人机的轨迹：得到 (pop_size, K, H+1, 2) 位置张量
        positions_per_uav = []
        for k in range(K):
            start_i, start_j = start_positions[k]
            dirs_k = all_dirs[:, k, :]  # (pop_size, H)
            pos_k = decode_dirs_torch(dirs_k, start_i, start_j, M, N)  # (pop_size, H+1, 2)
            positions_per_uav.append(pos_k)
        all_positions = torch.stack(positions_per_uav, dim=1)  # (pop_size, K, H+1, 2)

        # 4. 计算适应度：总奖励 = sum_{h=1..H} sum_{k} p_seq[h][pos]
        positions_h = all_positions[:, :, 1:, :]  # (pop_size, K, H, 2)
        idx_i = positions_h[:, :, :, 0] - 1  # 0-based
        idx_j = positions_h[:, :, :, 1] - 1
        flat_idx = idx_i * N + idx_j  # (pop_size, K, H)
        rewards = torch.zeros((pop_size, K, H), dtype=torch.float32, device=DEVICE)
        for h in range(H):
            p_h = p_seq_stack[h+1]  # (M*N,)
            idx_h = flat_idx[:, :, h]  # (pop_size, K)
            # 对每个无人机收集概率
            for k in range(K):
                idx_k = idx_h[:, k]  # (pop_size,)
                rewards[:, k, h] = p_h[idx_k]
        total_reward = rewards.sum(dim=(1,2))  # (pop_size,)

        # 5. 计算碰撞惩罚（简单双重循环，K 小，可接受）
        penalty = torch.zeros(pop_size, device=DEVICE)
        for h in range(H):
            pos_h = positions_h[:, :, h, :]  # (pop_size, K, 2)
            for i in range(pop_size):
                seen = set()
                for k in range(K):
                    coord = (pos_h[i, k, 0].item(), pos_h[i, k, 1].item())
                    if coord in seen:
                        penalty[i] += penalty_coef
                        break
                    seen.add(coord)
        fitness = total_reward * Pd - penalty
        return fitness.cpu().numpy()

    def optimize(self, start_positions, p_curr, penalty_coef):
        """
        start_positions: list of (i,j) 1-based，长度为 K
        p_curr: numpy array (M, N) 当前概率分布
        """
        p_curr_gpu = torch.from_numpy(p_curr).float().to(DEVICE)

        # 初始化评估
        for sidx, sub in enumerate(self.subpop):
            fitness = self.evaluate_batch_gpu(sidx, start_positions, p_curr_gpu, penalty_coef)
            for i in range(self.pop_size):
                if fitness[i] > sub['pbest_fit'][i]:
                    sub['pbest_fit'][i] = fitness[i]
                    sub['pbest'][i] = sub['particles'][i].copy()
                    sub['pbest_gpu'][i] = torch.from_numpy(sub['pbest'][i]).int().to(DEVICE)
                if fitness[i] > sub['gbest_fit']:
                    sub['gbest_fit'] = fitness[i]
                    sub['gbest'] = sub['particles'][i].copy()
                    sub['gbest_gpu'] = torch.from_numpy(sub['gbest']).int().to(DEVICE)

        # 迭代优化
        for it in range(self.max_iter):
            w = self.w_start - (self.w_start - self.w_end) * it / self.max_iter
            for sidx, sub in enumerate(self.subpop):
                r1 = torch.rand(self.pop_size, self.H, device=DEVICE)
                r2 = torch.rand(self.pop_size, self.H, device=DEVICE)
                vel = (w * sub['velocities_gpu'] +
                       self.c1 * r1 * (sub['pbest_gpu'] - sub['particles_gpu']) +
                       self.c2 * r2 * (sub['gbest_gpu'].unsqueeze(0) - sub['particles_gpu']))
                sub['velocities_gpu'] = vel
                new_p = sub['particles_gpu'] + vel
                new_p = torch.clamp(new_p, 0, 3)
                new_p = torch.round(new_p).int()
                sub['particles_gpu'] = new_p
                sub['particles'] = new_p.cpu().numpy()
                sub['velocities'] = vel.cpu().numpy()

                fitness = self.evaluate_batch_gpu(sidx, start_positions, p_curr_gpu, penalty_coef)
                for i in range(self.pop_size):
                    if fitness[i] > sub['pbest_fit'][i]:
                        sub['pbest_fit'][i] = fitness[i]
                        sub['pbest'][i] = sub['particles'][i].copy()
                        sub['pbest_gpu'][i] = torch.from_numpy(sub['pbest'][i]).int().to(DEVICE)
                    if fitness[i] > sub['gbest_fit']:
                        sub['gbest_fit'] = fitness[i]
                        sub['gbest'] = sub['particles'][i].copy()
                        sub['gbest_gpu'] = torch.from_numpy(sub['gbest']).int().to(DEVICE)

        return [sub['gbest'] for sub in self.subpop]

# ======================== 单次仿真（支持任意 K） ========================
def run_single_simulation(K, target_init_pos, uav_start_positions, params):
    pop_size = params['pop_size']
    max_iter = params['max_iter']
    penalty_coef = params['penalty_coef']
    coop_gap = params['coop_gap']
    w_end = params['w_end']

    p = np.ones((M, N), dtype=np.float32) / (M * N)
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
        optimizer = CC_MPSO_GPU(K, horizon, pop_size, max_iter, coop_gap,
                                w_start, w_end, c1, c2)
        start_positions = [(uav_pos_array[k][0], uav_pos_array[k][1]) for k in range(K)]
        best_dirs = optimizer.optimize(start_positions, p, penalty_coef)
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
            p_flat_gpu = torch.from_numpy(p_flat).float().to(DEVICE)
            p_flat_next = torch.mv(T_mat_gpu, p_flat_gpu)
            p = p_flat_next.cpu().numpy().reshape((M, N))
    return discovery_step if detected else max_steps

# ======================== 批量运行单个配置（支持断点续跑，支持指定 K） ========================
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
# 要扫描的无人机数量范围
K_LIST = list(range(2, 11))  # 2,3,4,5,6,7,8,9,10
# 每组仿真次数
TARGET_SIMS = 500

# 定义参数扫描列表（可根据需要修改，这里保留原12组）
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

if __name__ == "__main__":
    print("预热 GPU...")
    dummy = torch.mv(T_mat_gpu, torch.ones(M*N, device=DEVICE))
    print("预热完成。")

    print(f"\n将依次运行 K = {K_LIST} 的仿真，每组 {TARGET_SIMS} 次，共 {len(K_LIST)} 组。")
    print("每组将扫描以下参数组合（共12组）：")
    for idx, p in enumerate(PARAM_SWEEP):
        print(f"  {idx}: {p}")

    # 选择要运行的 K 范围（可以手动选择，这里默认全部）
    sel = input(f"\n请输入要运行的 K 值，多个用逗号分隔 (默认全部 {K_LIST}): ")
    if sel.strip():
        k_list = [int(x.strip()) for x in sel.split(',')]
    else:
        k_list = K_LIST

    # 选择要运行的参数组合索引（默认全部）
    param_sel = input(f"请输入要运行的参数组合索引，多个用逗号分隔 (默认全部 0-{len(PARAM_SWEEP)-1}): ")
    if param_sel.strip():
        param_indices = [int(x.strip()) for x in param_sel.split(',')]
    else:
        param_indices = list(range(len(PARAM_SWEEP)))

    for K in k_list:
        print(f"\n{'='*60}")
        print(f"开始处理 K = {K}")
        print(f"{'='*60}")
        for idx in param_indices:
            params = PARAM_SWEEP[idx]
            # 生成结果文件名，包含 K 和参数信息
            fname = f"results_K{K}_pop{params['pop_size']}_iter{params['max_iter']}_pen{params['penalty_coef']}_coop{params['coop_gap']}_wend{params['w_end']}.json"
            print(f"\n运行组合 K={K}, param_idx={idx}: {params}")
            run_config(params, TARGET_SIMS, fname, K)
        print(f"K={K} 的所有参数组合运行完毕！")

    print("\n所有指定 K 和参数组合运行完毕！")