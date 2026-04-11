# uav_simulator.py
# -*- coding: utf-8 -*-
"""
无人机协同搜索模拟器 - PyTorch CUDA加速版（修复版）
修复：PSO离散化、目标移动概率更新、多进程CUDA冲突
"""

import numpy as np
import random
import time
import torch
from multiprocessing import Pool, cpu_count
import logging
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')

# ======================== 检查CUDA ========================
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')

if CUDA_AVAILABLE:
    print(f"PyTorch CUDA可用: {CUDA_AVAILABLE}")
    print(f"使用设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")

# ======================== 参数设置 ========================
M, N = 16, 23  # 网格数
dx, dy = 20.0, 20.0
V_uav = 180.0
dt = dx / V_uav  # 时间步长 (h)
max_flight_time = 40.0
max_steps = int(max_flight_time / dt)  # 约360步
start_step = 23  # 第23步开始探测

Pd = 0.98  # 探测概率
Pf = 0.02  # 虚警概率
p_stay = 0.95  # 留在原网格概率
p_move = 0.05  # 移动到邻域总概率


class TransitionMatrix:
    """转移矩阵管理器（GPU加速）"""

    def __init__(self):
        self.T_mat = self._build_transition_matrix()
        if CUDA_AVAILABLE:
            self.T_mat = self.T_mat.to(DEVICE)

    def _build_transition_matrix(self):
        """构建转移矩阵"""
        n_cells = M * N
        T = torch.zeros(n_cells, n_cells, dtype=torch.float32)

        for idx in range(n_cells):
            i = idx // N
            j = idx % N
            T[idx, idx] = p_stay

            # 获取邻居
            neighbors = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < M and 0 <= nj < N:
                    neighbors.append(ni * N + nj)

            if neighbors:
                prob = p_move / len(neighbors)
                for nidx in neighbors:
                    T[nidx, idx] = prob

        return T

    def predict(self, p_flat):
        """概率预测"""
        if isinstance(p_flat, np.ndarray):
            p_flat = torch.from_numpy(p_flat).float().to(DEVICE)

        result = self.T_mat @ p_flat

        if CUDA_AVAILABLE:
            return result.cpu().numpy()
        else:
            return result.numpy()


class DiscretePSO:
    """离散PSO优化器（修复版）"""

    def __init__(self, K, H, pop_size, max_iter, w_start, w_end, c1, c2, penalty_coef):
        self.K = K
        self.H = H
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.penalty_coef = penalty_coef

        # 初始化粒子群（离散位置0-3）
        self.particles = np.random.randint(0, 4, size=(pop_size, K, H))
        # 速度范围[-4, 4]
        self.velocities = np.random.uniform(-3, 3, size=(pop_size, K, H))
        self.pbest = self.particles.copy()
        self.pbest_fit = np.full(pop_size, -np.inf)
        self.gbest = self.particles[0].copy()
        self.gbest_fit = -np.inf

    def _sigmoid(self, x):
        """sigmoid函数，用于将速度转换为概率"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _update_position(self, velocity, current_position):
        """使用概率方式更新离散位置"""
        prob = self._sigmoid(velocity)
        # 以概率prob改变方向，否则保持原方向
        change_mask = np.random.random(prob.shape) < prob
        new_directions = np.random.randint(0, 4, size=prob.shape)
        return np.where(change_mask, new_directions, current_position)

    def evaluate_fitness(self, start_positions, p_curr, transition):
        """评估适应度（修复版：正确处理概率预测）"""
        # 预测未来H步的概率分布
        p_seq = [p_curr.copy()]
        current_p = p_curr.copy()

        for _ in range(self.H):
            p_flat = current_p.flatten()
            p_next = transition.predict(p_flat)
            current_p = p_next.reshape((M, N))
            p_seq.append(current_p.copy())

        fitness = np.zeros(self.pop_size)

        for idx in range(self.pop_size):
            total_reward = 0.0
            penalty = 0.0

            for h in range(self.H):
                positions = []
                # 解码每个无人机的路径
                for k in range(self.K):
                    ci, cj = start_positions[k]
                    for step in range(h + 1):
                        if step < self.H:
                            d = self.particles[idx, k, step]
                            if d == 0:  # 上
                                ci = max(0, ci - 1)
                            elif d == 1:  # 下
                                ci = min(M - 1, ci + 1)
                            elif d == 2:  # 左
                                cj = max(0, cj - 1)
                            elif d == 3:  # 右
                                cj = min(N - 1, cj + 1)
                    positions.append((ci, cj))

                # 碰撞惩罚
                if len(set(positions)) != len(positions):
                    penalty += self.penalty_coef * (len(positions) - len(set(positions)))

                # 计算奖励（使用预测的概率）
                p_h = p_seq[h + 1]
                for k, (i, j) in enumerate(positions):
                    total_reward += p_h[i, j]

            # 适应度 = 期望发现概率 - 碰撞惩罚
            fitness[idx] = total_reward * Pd - penalty

        return fitness

    def optimize(self, start_positions, p_curr, transition):
        """优化主循环（修复版）"""
        # 初始化评估
        fitness = self.evaluate_fitness(start_positions, p_curr, transition)

        for i in range(self.pop_size):
            if fitness[i] > self.pbest_fit[i]:
                self.pbest_fit[i] = fitness[i]
                self.pbest[i] = self.particles[i].copy()
            if fitness[i] > self.gbest_fit:
                self.gbest_fit = fitness[i]
                self.gbest = self.particles[i].copy()

        # 迭代优化
        for it in range(self.max_iter):
            # 动态惯性权重
            w = self.w_start - (self.w_start - self.w_end) * it / self.max_iter

            for i in range(self.pop_size):
                for k in range(self.K):
                    for h in range(self.H):
                        r1, r2 = random.random(), random.random()

                        # 速度更新公式
                        vel = (w * self.velocities[i, k, h] +
                               self.c1 * r1 * (self.pbest[i, k, h] - self.particles[i, k, h]) +
                               self.c2 * r2 * (self.gbest[k, h] - self.particles[i, k, h]))

                        # 限制速度范围
                        self.velocities[i, k, h] = np.clip(vel, -4, 4)

                # 更新位置
                self.particles[i] = self._update_position(self.velocities[i], self.particles[i])

            # 评估新位置
            fitness = self.evaluate_fitness(start_positions, p_curr, transition)

            for i in range(self.pop_size):
                if fitness[i] > self.pbest_fit[i]:
                    self.pbest_fit[i] = fitness[i]
                    self.pbest[i] = self.particles[i].copy()
                if fitness[i] > self.gbest_fit:
                    self.gbest_fit = fitness[i]
                    self.gbest = self.particles[i].copy()

        return self.gbest


class GreedyPathOptimizer:
    """贪心路径优化器（快速版本）"""

    def __init__(self, K, H, penalty_coef=1000.0):
        self.K = K
        self.H = H
        self.penalty_coef = penalty_coef

    def optimize(self, uav_positions, p_grid):
        """贪心优化：每步选择概率最高的方向"""
        best_dirs = np.zeros((self.K, self.H), dtype=np.int32)
        current_pos = uav_positions.copy()

        for step in range(self.H):
            for k in range(self.K):
                ci, cj = current_pos[k]

                # 评估四个方向的奖励
                rewards = []
                for d in range(4):
                    ni, nj = ci, cj
                    if d == 0:  # 上
                        ni = max(0, ci - 1)
                    elif d == 1:  # 下
                        ni = min(M - 1, ci + 1)
                    elif d == 2:  # 左
                        nj = max(0, cj - 1)
                    elif d == 3:  # 右
                        nj = min(N - 1, cj + 1)

                    rewards.append(p_grid[ni, nj])

                # 选择最佳方向
                best_dirs[k, step] = np.argmax(rewards)

                # 更新位置
                d = best_dirs[k, step]
                if d == 0:
                    current_pos[k] = (max(0, ci - 1), cj)
                elif d == 1:
                    current_pos[k] = (min(M - 1, ci + 1), cj)
                elif d == 2:
                    current_pos[k] = (ci, max(0, cj - 1))
                elif d == 3:
                    current_pos[k] = (ci, min(N - 1, cj + 1))

        return best_dirs


class UAVSimulatorCUDA:
    """CUDA加速的无人机模拟器（修复版）"""

    def __init__(self, use_cuda=True):
        """
        初始化模拟器
        Args:
            use_cuda: 是否使用CUDA（多进程时应设为False）
        """
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.transition = TransitionMatrix()
        self.device_info = self._get_device_info()

    def _get_device_info(self):
        """获取设备信息"""
        info = {
            'cuda_available': CUDA_AVAILABLE,
            'use_cuda': self.use_cuda,
            'device_name': torch.cuda.get_device_name(0) if self.use_cuda else 'CPU',
            'cuda_version': torch.version.cuda if CUDA_AVAILABLE else 'N/A'
        }
        return info

    def _move_target(self, target_i, target_j, p):
        """
        移动目标并更新概率地图
        Args:
            target_i, target_j: 当前目标位置
            p: 当前概率地图
        Returns:
            新目标位置，更新后的概率地图
        """
        # 收集邻居
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = target_i + di, target_j + dj
            if 0 <= ni < M and 0 <= nj < N:
                neighbors.append((ni, nj))

        if not neighbors:
            return target_i, target_j, p

        # 选择新位置
        new_i, new_j = random.choice(neighbors)

        # 更新概率地图：将概率从原位置转移到新位置
        prob_transfer = p[target_i, target_j] * p_move
        p[target_i, target_j] -= prob_transfer
        p[new_i, new_j] += prob_transfer

        # 确保概率不为负
        p = np.maximum(p, 0)
        p_sum = np.sum(p)
        if p_sum > 0:
            p /= p_sum

        return new_i, new_j, p

    def run_simulation(self, K, target_init_pos, uav_start_positions, params):
        """运行单次仿真（修复版）"""
        # 转换坐标（从1-based到0-based）
        target_i, target_j = target_init_pos[0] - 1, target_init_pos[1] - 1
        uav_pos = np.array([(p[0] - 1, p[1] - 1) for p in uav_start_positions], dtype=np.int32)

        # 初始化概率地图
        p = np.ones((M, N), dtype=np.float32) / (M * N)
        total_steps_done = start_step - 1
        detected = False
        discovery_step = None

        # 获取参数
        H = params.get('H', 6)
        use_pso = params.get('use_pso', True)

        while total_steps_done < max_steps and not detected:
            remaining = max_steps - total_steps_done
            horizon = min(H, remaining)

            if horizon <= 0:
                break

            # 根据参数选择优化器
            if use_pso and all(k in params for k in ['pop_size', 'max_iter', 'w_start', 'w_end', 'c1', 'c2']):
                # 使用PSO优化器
                optimizer = DiscretePSO(
                    K, horizon,
                    params.get('pop_size', 12),
                    params.get('max_iter', 20),
                    params.get('w_start', 0.9),
                    params.get('w_end', 0.4),
                    params.get('c1', 1.5),
                    params.get('c2', 1.5),
                    params.get('penalty_coef', 500.0)
                )
                best_dirs = optimizer.optimize(uav_pos, p, self.transition)
            else:
                # 使用贪心优化器
                optimizer = GreedyPathOptimizer(K, horizon, params.get('penalty_coef', 500.0))
                best_dirs = optimizer.optimize(uav_pos, p)

            # 执行路径
            for step in range(horizon):
                if total_steps_done >= max_steps or detected:
                    break

                # 移动无人机
                new_positions = []
                for k in range(K):
                    d = best_dirs[k, step]
                    ci, cj = uav_pos[k]
                    if d == 0:  # 上
                        ci = max(0, ci - 1)
                    elif d == 1:  # 下
                        ci = min(M - 1, ci + 1)
                    elif d == 2:  # 左
                        cj = max(0, cj - 1)
                    elif d == 3:  # 右
                        cj = min(N - 1, cj + 1)
                    new_positions.append((ci, cj))

                # 探测更新
                for k, (ci, cj) in enumerate(new_positions):
                    # 生成观测
                    if target_i == ci and target_j == cj:
                        obs = random.random() < Pd
                    else:
                        obs = random.random() < Pf

                    # 贝叶斯更新
                    prior = p[ci, cj]
                    if obs:
                        p[ci, cj] = (prior * Pd) / (prior * Pd + (1 - prior) * Pf)
                    else:
                        p[ci, cj] = (prior * (1 - Pd)) / (prior * (1 - Pd) + (1 - prior) * (1 - Pf))

                    # 归一化
                    p_sum = np.sum(p)
                    if p_sum > 0:
                        p /= p_sum

                    if obs and target_i == ci and target_j == cj:
                        detected = True
                        discovery_step = total_steps_done + 1
                        break

                if detected:
                    break

                # 更新状态
                uav_pos = np.array(new_positions, dtype=np.int32)
                total_steps_done += 1

                # 移动目标（修复版：更新概率地图）
                if random.random() >= p_stay:
                    target_i, target_j, p = self._move_target(target_i, target_j, p)

                # GPU加速的概率预测
                p_flat = p.flatten()
                p_flat = self.transition.predict(p_flat)
                p = p_flat.reshape((M, N))

        # 转换回1-based坐标（保持兼容性）
        return discovery_step if detected else max_steps

    def batch_simulate_serial(self, K, init_positions, uav_start, params):
        """串行批量仿真（可使用CUDA）"""
        detect_steps = []
        for init_pos in init_positions:
            step = self.run_simulation(K, init_pos, uav_start, params)
            detect_steps.append(step)
        return np.array(detect_steps)

    def batch_simulate_parallel(self, K, init_positions, uav_start, params, n_processes=None):
        """并行批量仿真（强制使用CPU）"""
        if n_processes is None:
            n_processes = min(cpu_count(), 4)

        # 准备参数列表
        args_list = [(K, init_pos, uav_start, params) for init_pos in init_positions]

        with Pool(processes=n_processes) as pool:
            detect_steps = list(pool.starmap(_run_simulation_cpu, args_list))

        return np.array(detect_steps)

    def batch_simulate(self, K, init_positions, uav_start, params, n_processes=1):
        """
        批量仿真（自动选择模式）
        Args:
            n_processes: 进程数，1表示串行（可使用CUDA），>1表示并行（强制CPU）
        """
        if n_processes == 1:
            return self.batch_simulate_serial(K, init_positions, uav_start, params)
        else:
            return self.batch_simulate_parallel(K, init_positions, uav_start, params, n_processes)


def _run_simulation_cpu(K, target_init_pos, uav_start_positions, params):
    """工作进程的仿真函数（强制CPU模式）"""
    sim = UAVSimulatorCUDA(use_cuda=False)
    return sim.run_simulation(K, target_init_pos, uav_start_positions, params)


def quick_test():
    """快速测试"""
    print("初始化模拟器...")
    simulator = UAVSimulatorCUDA()

    # 测试参数
    K = 7
    params = {
        'H': 6,
        'use_pso': False,
        'penalty_coef': 500.0
    }

    uav_start = [(1, 1)] * K
    init_positions = [(random.randint(1, M), random.randint(1, N)) for _ in range(10)]

    print("运行测试仿真...")
    start_time = time.time()

    detect_steps = simulator.batch_simulate(K, init_positions, uav_start, params, n_processes=1)

    elapsed = time.time() - start_time
    detection_rate = np.sum(detect_steps < max_steps) / len(detect_steps) * 100

    print(f"\n测试完成!")
    print(f"  仿真次数: {len(init_positions)}")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  平均发现时间: {np.mean(detect_steps) * dt:.2f} 小时")
    print(f"  发现率: {detection_rate:.1f}%")

    return simulator


if __name__ == "__main__":
    quick_test()