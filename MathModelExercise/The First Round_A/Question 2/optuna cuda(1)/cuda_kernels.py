# -*- coding: utf-8 -*-
"""
CUDA加速内核函数
"""

import numpy as np
from numba import cuda, float32, int32, boolean
import math


# ======================== CUDA线程配置 ========================
@cuda.jit
def predict_probability_cuda(p_flat, T_mat, result):
    """GPU并行概率预测"""
    idx = cuda.grid(1)
    if idx < p_flat.shape[0]:
        total = 0.0
        for j in range(T_mat.shape[1]):
            total += T_mat[idx, j] * p_flat[j]
        result[idx] = total


@cuda.jit
def bayesian_update_cuda(p_grid, detected, i, j, Pd, Pf, M, N):
    """GPU并行贝叶斯更新"""
    idx = cuda.grid(1)
    if idx == 0:
        prior = p_grid[i, j]
        if detected:
            post = (prior * Pd) / (prior * Pd + (1 - prior) * Pf)
        else:
            post = (prior * (1 - Pd)) / (prior * (1 - Pd) + (1 - prior) * (1 - Pf))
        p_grid[i, j] = post

        # 归一化
        total = 0.0
        for x in range(M):
            for y in range(N):
                total += p_grid[x, y]
        if total > 0:
            for x in range(M):
                for y in range(N):
                    p_grid[x, y] /= total


@cuda.jit
def batch_predict_probability(p_flat_batch, T_mat, results_batch):
    """批量概率预测"""
    batch_idx = cuda.blockIdx.x
    thread_idx = cuda.threadIdx.x

    if thread_idx < p_flat_batch.shape[1]:
        total = 0.0
        for j in range(T_mat.shape[1]):
            total += T_mat[thread_idx, j] * p_flat_batch[batch_idx, j]
        results_batch[batch_idx, thread_idx] = total


@cuda.jit
def evaluate_path_cuda(uav_positions, p_seq, rewards, K, H, M, N, Pd):
    """GPU并行评估所有路径的奖励"""
    idx = cuda.grid(1)
    if idx < rewards.shape[0]:
        total_reward = 0.0
        for h in range(H):
            for k in range(K):
                i = uav_positions[idx, k, h, 0]
                j = uav_positions[idx, k, h, 1]
                if 0 <= i < M and 0 <= j < N:
                    total_reward += p_seq[h, i, j]
        rewards[idx] = total_reward * Pd


@cuda.jit
def update_particles_cuda(particles, velocities, pbest, gbest,
                          w, c1, c2, pop_size, H):
    """GPU并行更新粒子位置和速度"""
    idx = cuda.grid(1)
    if idx < pop_size:
        for h in range(H):
            r1 = cuda.random.xoroshiro128p_uniform_float32()
            r2 = cuda.random.xoroshiro128p_uniform_float32()

            vel = w * velocities[idx, h] + \
                  c1 * r1 * (pbest[idx, h] - particles[idx, h]) + \
                  c2 * r2 * (gbest[h] - particles[idx, h])

            velocities[idx, h] = vel
            new_pos = particles[idx, h] + vel
            particles[idx, h] = max(0, min(3, int(round(new_pos))))


@cuda.jit
def calculate_coverage_cuda(p_grid, uav_positions, K, coverage):
    """GPU并行计算覆盖度"""
    idx = cuda.grid(1)
    if idx < K:
        i = uav_positions[idx, 0]
        j = uav_positions[idx, 1]
        if 0 <= i < p_grid.shape[0] and 0 <= j < p_grid.shape[1]:
            coverage[idx] = p_grid[i, j]


@cuda.jit
def fast_movement_cuda(uav_positions, directions, new_positions, K, H, M, N):
    """GPU并行计算无人机移动"""
    idx = cuda.grid(1)
    if idx < K:
        for h in range(H):
            ci = uav_positions[idx, 0]
            cj = uav_positions[idx, 1]
            d = directions[idx, h]

            if d == 0:  # 上
                ci = max(0, ci - 1)
            elif d == 1:  # 下
                ci = min(M - 1, ci + 1)
            elif d == 2:  # 左
                cj = max(0, cj - 1)
            elif d == 3:  # 右
                cj = min(N - 1, cj + 1)

            new_positions[idx, h, 0] = ci
            new_positions[idx, h, 1] = cj


class CUDAAccelerator:
    """CUDA加速器管理类"""

    def __init__(self):
        self.stream = cuda.stream()
        self.gpu_memory = {}

    def to_gpu(self, data, name):
        """将数据转移到GPU"""
        if name in self.gpu_memory:
            del self.gpu_memory[name]
        self.gpu_memory[name] = cuda.to_device(data, stream=self.stream)
        return self.gpu_memory[name]

    def from_gpu(self, name):
        """从GPU获取数据"""
        if name in self.gpu_memory:
            return self.gpu_memory[name].copy_to_host()
        return None

    def synchronize(self):
        """同步GPU"""
        self.stream.synchronize()

    def clear_memory(self):
        """清理GPU内存"""
        for key in list(self.gpu_memory.keys()):
            del self.gpu_memory[key]
        cuda.current_context().get_memory_info()