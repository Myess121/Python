# uav_simulator.py
# -*- coding: utf-8 -*-
"""
CC-MPSO无人机协同搜索模拟器（Numba加速版）
"""

import numpy as np
import random
import time
from numba import jit
from multiprocessing import Pool
import warnings

warnings.filterwarnings('ignore')

# 从config导入参数
from config import (
    M, N, dx, dy, V_uav, dt, max_flight_time, max_steps, start_step,
    Pd, Pf, p_stay, p_move, DEFAULT_CCMPSO_PARAMS
)


# ======================== Numba加速的核心函数 ========================
@jit(nopython=True)
def get_neighbors_numba(i, j, M, N):
    """获取邻居网格（1-based坐标）"""
    neighbors = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 1 <= ni <= M and 1 <= nj <= N:
            neighbors.append((ni, nj))
    return neighbors


@jit(nopython=True)
def build_transition_matrix_numba(M, N, p_stay, p_move):
    """构建转移矩阵"""
    n_cells = M * N
    T = np.zeros((n_cells, n_cells))

    for idx in range(n_cells):
        i = idx // N + 1
        j = idx % N + 1
        row_idx = (i - 1) * N + (j - 1)
        T[row_idx, idx] = p_stay

        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 1 <= ni <= M and 1 <= nj <= N:
                neighbors.append((ni, nj))

        if len(neighbors) > 0:
            prob = p_move / len(neighbors)
            for ni, nj in neighbors:
                row_idx2 = (ni - 1) * N + (nj - 1)
                T[row_idx2, idx] = prob

    return T


@jit(nopython=True)
def predict_probability_numba(p_flat, T_mat):
    """概率预测"""
    return T_mat @ p_flat


@jit(nopython=True)
def bayesian_update_numba(p_grid, detected, i, j, Pd, Pf):
    """贝叶斯更新"""
    prior = p_grid[i - 1, j - 1]
    if detected:
        post = (prior * Pd) / (prior * Pd + (1 - prior) * Pf)
    else:
        post = (prior * (1 - Pd)) / (prior * (1 - Pd) + (1 - prior) * (1 - Pf))
    p_grid[i - 1, j - 1] = post

    # 归一化
    total = np.sum(p_grid)
    if total > 0:
        p_grid /= total
    return p_grid


@jit(nopython=True)
def move_target_numba(i, j, p_stay, p_move, M, N):
    """目标移动"""
    if np.random.random() < p_stay:
        return i, j

    neighbors = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 1 <= ni <= M and 1 <= nj <= N:
            neighbors.append((ni, nj))

    if len(neighbors) > 0:
        idx = np.random.randint(0, len(neighbors))
        return neighbors[idx]
    return i, j


@jit(nopython=True)
def generate_observation_numba(true_i, true_j, probe_i, probe_j, Pd, Pf):
    """生成观测值"""
    if true_i == probe_i and true_j == probe_j:
        return np.random.random() < Pd
    else:
        return np.random.random() < Pf


@jit(nopython=True)
def decode_dirs_numba(dirs, start_i, start_j, M, N):
    """解码方向序列为轨迹"""
    traj_i = np.zeros(len(dirs) + 1, dtype=np.int32)
    traj_j = np.zeros(len(dirs) + 1, dtype=np.int32)
    traj_i[0] = start_i
    traj_j[0] = start_j

    ci, cj = start_i, start_j
    for idx, d in enumerate(dirs):
        if d == 0:  # 上
            ci = max(1, ci - 1)
        elif d == 1:  # 下
            ci = min(M, ci + 1)
        elif d == 2:  # 左
            cj = max(1, cj - 1)
        elif d == 3:  # 右
            cj = min(N, cj + 1)
        traj_i[idx + 1] = ci
        traj_j[idx + 1] = cj

    return traj_i, traj_j


# ======================== 预计算转移矩阵 ========================
T_mat = build_transition_matrix_numba(M, N, p_stay, p_move)


# ======================== CC-MPSO 类 ========================
class CC_MPSO:
    """
    协同进化多粒子群优化器 (Cooperative Co-evolutionary Multi-swarm PSO)
    每个无人机拥有独立的子种群，子种群之间协同进化
    """

    def __init__(self, K, H, pop_size, max_iter, coop_gap, w_start, w_end, c1, c2):
        """
        初始化CC-MPSO

        参数:
            K: 无人机数量
            H: 规划步数
            pop_size: 每个子种群的粒子数
            max_iter: 最大迭代次数
            coop_gap: 协同评估间隔
            w_start: 惯性权重起始值
            w_end: 惯性权重结束值
            c1: 认知系数
            c2: 社会系数
        """
        self.K = K
        self.H = H
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.coop_gap = coop_gap
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2

        # 初始化K个子种群
        self.subpop = []
        for _ in range(K):
            # 粒子位置：0-3表示四个方向
            particles = np.random.randint(0, 4, size=(pop_size, H))
            # 速度
            velocities = np.random.uniform(-1, 1, size=(pop_size, H))
            # 个体最优
            pbest = particles.copy()
            pbest_fit = np.full(pop_size, -np.inf)
            # 全局最优
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

    def evaluate_fitness(self, all_dirs, start_positions, p_curr, penalty_coef):
        """
        评估适应度

        参数:
            all_dirs: K个无人机的方向序列
            start_positions: 起始位置列表
            p_curr: 当前概率地图
            penalty_coef: 碰撞惩罚系数

        返回:
            适应度值
        """
        # 预测未来H步的概率分布
        p_seq = [p_curr.flatten()]
        for _ in range(self.H):
            p_seq.append(predict_probability_numba(p_seq[-1], T_mat))

        total_reward = 0.0
        penalty = 0.0

        for h in range(self.H):
            positions = []
            for k in range(self.K):
                traj_i, traj_j = decode_dirs_numba(
                    all_dirs[k],
                    start_positions[k][0],
                    start_positions[k][1],
                    M, N
                )
                positions.append((traj_i[h + 1], traj_j[h + 1]))

            # 碰撞检测与惩罚
            unique_pos = set(positions)
            if len(unique_pos) != len(positions):
                penalty += penalty_coef

            # 计算期望奖励
            p_flat = p_seq[h + 1]
            for k, (i, j) in enumerate(positions):
                idx = (i - 1) * N + (j - 1)
                total_reward += p_flat[idx]

        return total_reward * Pd - penalty

    def optimize(self, start_positions, p_curr, penalty_coef):
        """
        执行CC-MPSO优化

        返回:
            最优方向序列列表 (K × H)
        """
        # ========== 初始化评估 ==========
        for sidx, sub in enumerate(self.subpop):
            for i in range(self.pop_size):
                dirs = sub['particles'][i]
                all_dirs = []
                for k in range(self.K):
                    if k == sidx:
                        all_dirs.append(dirs)
                    else:
                        all_dirs.append(self.subpop[k]['gbest'])

                fit = self.evaluate_fitness(all_dirs, start_positions, p_curr, penalty_coef)

                if fit > sub['pbest_fit'][i]:
                    sub['pbest_fit'][i] = fit
                    sub['pbest'][i] = dirs.copy()
                if fit > sub['gbest_fit']:
                    sub['gbest_fit'] = fit
                    sub['gbest'] = dirs.copy()

        # ========== 迭代优化 ==========
        for it in range(self.max_iter):
            # 动态惯性权重
            w = self.w_start - (self.w_start - self.w_end) * it / self.max_iter

            for sidx, sub in enumerate(self.subpop):
                # 更新速度和位置
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

                # 协同评估
                if (it + 1) % self.coop_gap == 0:
                    for i in range(self.pop_size):
                        dirs = sub['particles'][i]
                        all_dirs = []
                        for k in range(self.K):
                            if k == sidx:
                                all_dirs.append(dirs)
                            else:
                                all_dirs.append(self.subpop[k]['gbest'])

                        fit = self.evaluate_fitness(all_dirs, start_positions, p_curr, penalty_coef)

                        if fit > sub['pbest_fit'][i]:
                            sub['pbest_fit'][i] = fit
                            sub['pbest'][i] = dirs.copy()
                        if fit > sub['gbest_fit']:
                            sub['gbest_fit'] = fit
                            sub['gbest'] = dirs.copy()

        return [sub['gbest'] for sub in self.subpop]


# ======================== 无人机模拟器 ========================
class UAVSimulator:
    """CC-MPSO无人机协同搜索模拟器"""

    def __init__(self):
        """初始化模拟器"""
        self.params = DEFAULT_CCMPSO_PARAMS.copy()

    def set_params(self, **kwargs):
        """设置模拟器参数"""
        self.params.update(kwargs)

    def run_simulation(self, K, target_init_pos, uav_start_positions, params=None):
        """
        运行单次仿真

        参数:
            K: 无人机数量
            target_init_pos: 目标初始位置 (1-based, (i, j))
            uav_start_positions: 无人机起始位置列表
            params: 可选参数，覆盖默认参数

        返回:
            发现步数（若未发现则返回max_steps）
        """
        # 合并参数
        sim_params = self.params.copy()
        if params:
            sim_params.update(params)

        # 提取参数
        H = sim_params.get('H', 5)
        pop_size = sim_params.get('pop_size', 20)
        max_iter = sim_params.get('max_iter', 22)
        coop_gap = sim_params.get('coop_gap', 5)
        w_start = sim_params.get('w_start', 0.9075)
        w_end = sim_params.get('w_end', 0.2701)
        c1 = sim_params.get('c1', 1.8594)
        c2 = sim_params.get('c2', 1.1079)
        penalty_coef = sim_params.get('penalty_coef', 1974.16)

        # 初始化状态
        target_i, target_j = target_init_pos
        uav_pos = list(uav_start_positions)
        total_steps_done = start_step - 1

        # 初始化概率地图（均匀分布）
        p = np.ones((M, N)) / (M * N)
        detected = False
        discovery_step = None
        uav_pos_array = np.array(uav_pos, dtype=np.int32)

        while total_steps_done < max_steps and not detected:
            remaining = max_steps - total_steps_done
            horizon = min(H, remaining)
            if horizon <= 0:
                break

            # 创建CC-MPSO优化器
            optimizer = CC_MPSO(
                K, horizon, pop_size, max_iter, coop_gap,
                w_start, w_end, c1, c2
            )

            start_positions = [(uav_pos_array[k][0], uav_pos_array[k][1]) for k in range(K)]
            best_dirs = optimizer.optimize(start_positions, p, penalty_coef)

            # 执行规划周期
            for step in range(horizon):
                if total_steps_done >= max_steps or detected:
                    break

                # 移动无人机
                new_positions = []
                for k in range(K):
                    dirs = best_dirs[k]
                    d = dirs[step]
                    ci, cj = uav_pos_array[k]
                    if d == 0:
                        ci = max(1, ci - 1)
                    elif d == 1:
                        ci = min(M, ci + 1)
                    elif d == 2:
                        cj = max(1, cj - 1)
                    elif d == 3:
                        cj = min(N, cj + 1)
                    new_positions.append((ci, cj))

                # 探测与更新
                for k, (ci, cj) in enumerate(new_positions):
                    obs = generate_observation_numba(target_i, target_j, ci, cj, Pd, Pf)
                    p = bayesian_update_numba(p, obs, ci, cj, Pd, Pf)

                    if obs and ci == target_i and cj == target_j:
                        detected = True
                        discovery_step = total_steps_done + 1
                        break

                if detected:
                    break

                # 更新状态
                uav_pos_array = np.array(new_positions, dtype=np.int32)
                total_steps_done += 1

                # 移动目标
                target_i, target_j = move_target_numba(
                    target_i, target_j, p_stay, p_move, M, N
                )

                # 概率预测
                p_flat = p.flatten()
                p_flat = predict_probability_numba(p_flat, T_mat)
                p = p_flat.reshape((M, N))

        return discovery_step if detected else max_steps

    def batch_simulate(self, K, init_positions, uav_start, params=None, verbose=False):
        """
        批量仿真

        参数:
            K: 无人机数量
            init_positions: 初始目标位置列表
            uav_start: 无人机起始位置
            params: 可选参数
            verbose: 是否显示进度

        返回:
            发现步数数组
        """
        detect_steps = []
        total = len(init_positions)

        for idx, init_pos in enumerate(init_positions):
            if verbose and (idx + 1) % 100 == 0:
                print(f"    仿真进度: {idx + 1}/{total}")

            step = self.run_simulation(K, init_pos, uav_start, params)
            detect_steps.append(step)

        return np.array(detect_steps)

    def quick_test(self, K=7, N_sim=10):
        """快速测试"""
        print(f"\n快速测试: K={K}, N_sim={N_sim}")

        # 生成随机初始目标位置
        random.seed(42)
        init_positions = [(random.randint(1, M), random.randint(1, N)) for _ in range(N_sim)]
        uav_start = [(1, 1)] * K

        start_time = time.time()
        detect_steps = self.batch_simulate(K, init_positions, uav_start, verbose=True)
        elapsed = time.time() - start_time

        avg_time = np.mean(detect_steps) * dt
        detection_rate = np.sum(detect_steps < max_steps) / N_sim * 100
        rate_10h = np.sum(detect_steps < (10.0 / dt)) / N_sim * 100

        print(f"\n测试结果:")
        print(f"  平均发现时间: {avg_time:.2f} 小时")
        print(f"  40小时发现率: {detection_rate:.1f}%")
        print(f"  10小时发现率: {rate_10h:.1f}%")
        print(f"  耗时: {elapsed:.2f} 秒")

        return detect_steps


# ======================== 工作进程函数（用于多进程） ========================
def _run_simulation_worker(args):
    """多进程工作函数"""
    K, target_init_pos, uav_start, params = args
    sim = UAVSimulator()
    return sim.run_simulation(K, target_init_pos, uav_start, params)


# ======================== 测试入口 ========================
if __name__ == "__main__":
    print("=" * 60)
    print("CC-MPSO 无人机协同搜索模拟器")
    print("=" * 60)

    # 预热JIT编译
    print("预热Numba编译器...")
    _ = predict_probability_numba(np.ones(M * N), T_mat)
    print("初始化完成！")

    # 快速测试
    simulator = UAVSimulator()
    simulator.quick_test(K=7, N_sim=20)