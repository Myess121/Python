# -*- coding: utf-8 -*-
"""
BZK-005无人机协同搜索 - 带配置文件和训练日志版本
"""

import numpy as np
import random
import time
from multiprocessing import Pool, cpu_count
from numba import jit
import json
from datetime import datetime
import os
import logging
from pathlib import Path

# 导入配置文件
import config

# ======================== 参数设置（从配置文件读取）=======================
M, N = 16, 23  # 网格数 (纬度, 经度) - 固定参数，不放入配置
dx, dy = 20.0, 20.0  # 网格边长 (km) - 固定参数
V_uav = 180.0  # 无人机速度 (km/h) - 固定参数
dt = dx / V_uav  # 时间步长 (h)
max_flight_time = 40.0  # 最大续航 (h) - 固定参数
max_steps = int(max_flight_time / dt)
start_step = 23  # 第23步开始探测（前22步入场）- 固定参数

Pd = 0.95  # 探测概率 - 固定参数
Pf = 0.05  # 虚警概率 - 固定参数

# 目标运动参数 - 固定参数
p_stay = 0.95
p_move = 0.05

# 从配置文件读取的参数
H = config.H
pop_size = config.pop_size
max_iter = config.max_iter
coop_gap = config.coop_gap
w_start = config.w_start
w_end = config.w_end
c1 = config.c1
c2 = config.c2
penalty_coef = config.penalty_coef


# ======================== 设置日志系统 ========================
def setup_logging(config_name="default"):
    """设置训练日志"""
    # 创建日志目录
    log_dir = Path("training_logs")
    log_dir.mkdir(exist_ok=True)

    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{config_name}_{timestamp}.log"

    # 配置日志格式
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

    return log_file, timestamp


def log_config():
    """记录当前配置参数"""
    logging.info("=" * 80)
    logging.info("当前配置参数:")
    logging.info(f"  K_LIST = {config.K_LIST}")
    logging.info(f"  N_SIM_PER_K = {config.N_SIM_PER_K}")
    logging.info(f"  H = {config.H}")
    logging.info(f"  pop_size = {config.pop_size}")
    logging.info(f"  max_iter = {config.max_iter}")
    logging.info(f"  coop_gap = {config.coop_gap}")
    logging.info(f"  w_start = {config.w_start}, w_end = {config.w_end}")
    logging.info(f"  c1 = {config.c1}, c2 = {config.c2}")
    logging.info(f"  penalty_coef = {config.penalty_coef}")
    logging.info(f"  USE_ADAPTIVE_H = {config.USE_ADAPTIVE_H}")
    logging.info(f"  USE_EARLY_STOP = {config.USE_EARLY_STOP}")
    logging.info(f"  N_PROCESSES = {config.N_PROCESSES}")
    logging.info("=" * 80)


def save_training_summary(all_results, config_name, timestamp, log_file):
    """保存训练总结"""
    summary_dir = Path("training_summaries")
    summary_dir.mkdir(exist_ok=True)

    summary_file = summary_dir / f"summary_{config_name}_{timestamp}.json"

    summary = {
        'config_name': config_name,
        'timestamp': timestamp,
        'log_file': str(log_file),
        'config': {
            'K_LIST': config.K_LIST,
            'N_SIM_PER_K': config.N_SIM_PER_K,
            'H': config.H,
            'pop_size': config.pop_size,
            'max_iter': config.max_iter,
            'coop_gap': config.coop_gap,
            'w_start': config.w_start,
            'w_end': config.w_end,
            'c1': config.c1,
            'c2': config.c2,
            'penalty_coef': config.penalty_coef,
            'USE_ADAPTIVE_H': config.USE_ADAPTIVE_H,
            'USE_EARLY_STOP': config.USE_EARLY_STOP,
        },
        'results': all_results
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"\n训练总结已保存到: {summary_file}")
    return summary_file


# ======================== Numba加速的函数（保持不变）=======================
@jit(nopython=True)
def get_neighbors_numba(i, j, M, N):
    """获取邻居网格"""
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
    """解码方向序列"""
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


# ======================== CC-MPSO 优化版 ========================
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
        """快速评估函数"""
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
                positions.append((traj_i[h + 1], traj_j[h + 1]))

            unique_pos = set(positions)
            if len(unique_pos) != len(positions):
                penalty += penalty_coef

            p_flat = p_seq[h + 1]
            for k, (i, j) in enumerate(positions):
                idx = (i - 1) * N + (j - 1)
                total_reward += p_flat[idx]

        return total_reward * Pd - penalty

    def optimize(self, start_positions, p_curr, T_mat, M, N, Pd, penalty_coef):
        """优化主循环"""
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
                                         T_mat, M, N, Pd, penalty_coef)
                if fit > sub['pbest_fit'][i]:
                    sub['pbest_fit'][i] = fit
                    sub['pbest'][i] = dirs.copy()
                if fit > sub['gbest_fit']:
                    sub['gbest_fit'] = fit
                    sub['gbest'] = dirs.copy()

        # 迭代优化
        for it in range(self.max_iter):
            # 早停检查
            if config.USE_EARLY_STOP:
                prev_gbest_fit = sum([sub['gbest_fit'] for sub in self.subpop])

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

            # 早停判断
            if config.USE_EARLY_STOP:
                curr_gbest_fit = sum([sub['gbest_fit'] for sub in self.subpop])
                if it > 0 and abs(curr_gbest_fit - prev_gbest_fit) < 1e-6:
                    logging.debug(f"早停于迭代 {it}")
                    break

        return [sub['gbest'] for sub in self.subpop]


# ======================== 预计算转移矩阵 ========================
T_mat = build_transition_matrix_numba(M, N, p_stay, p_move)


# ======================== 无动画仿真 ========================
def run_simulation_no_animation_fast(K, target_init_pos, uav_start_positions):
    """单次仿真，返回发现时间步数"""
    p = np.ones((M, N)) / (M * N)
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


# ======================== 并行仿真包装函数 ========================
def run_simulation_wrapper(args):
    """多进程包装函数"""
    K, target_init_pos, uav_start_positions = args
    return run_simulation_no_animation_fast(K, target_init_pos, uav_start_positions)


# ======================== 批量测试函数 ========================
def test_single_K(K, N_sim, uav_start, n_processes):
    """测试单个K值"""
    logging.info(f"\n{'=' * 60}")
    logging.info(f"开始测试 K = {K}")
    logging.info(f"仿真次数: {N_sim}")
    logging.info(f"{'=' * 60}")

    random.seed(int(time.time()) + K)
    init_positions = [(random.randint(1, M), random.randint(1, N)) for _ in range(N_sim)]

    start_time = time.time()

    with Pool(processes=n_processes) as pool:
        args_list = [(K, init_pos, uav_start) for init_pos in init_positions]
        detect_steps = list(pool.map(run_simulation_wrapper, args_list))

    elapsed_time = time.time() - start_time

    detect_steps = np.array(detect_steps)
    avg_step = np.mean(detect_steps)
    std_step = np.std(detect_steps)
    avg_time = avg_step * dt
    std_time = std_step * dt

    confidence_95 = 1.96 * std_step / np.sqrt(N_sim)
    detection_rate = np.sum(detect_steps < max_steps) / N_sim * 100
    time_10h_steps = 10.0 / dt
    rate_10h = np.sum(detect_steps < time_10h_steps) / N_sim * 100

    results = {
        'K': K,
        'N_sim': N_sim,
        'avg_steps': float(avg_step),
        'std_steps': float(std_step),
        'avg_time_hours': float(avg_time),
        'std_time_hours': float(std_time),
        'ci_lower': float(avg_time - confidence_95 * dt),
        'ci_upper': float(avg_time + confidence_95 * dt),
        'detection_rate': detection_rate,
        'rate_within_10h': rate_10h,
        'elapsed_seconds': elapsed_time,
    }

    if config.SAVE_DETAILED_RESULTS:
        results['all_steps'] = detect_steps.tolist()

    logging.info(f"\nK={K} 测试完成!")
    logging.info(f"  耗时: {elapsed_time:.2f} 秒 ({elapsed_time / 3600:.2f} 小时)")
    logging.info(f"  平均发现时间: {avg_time:.2f} ± {std_time:.2f} 小时")
    logging.info(f"  95% 置信区间: [{avg_time - confidence_95 * dt:.2f}, {avg_time + confidence_95 * dt:.2f}] 小时")
    logging.info(f"  40小时内发现率: {detection_rate:.1f}%")
    logging.info(f"  10小时内发现率: {rate_10h:.1f}%")

    return results


# ======================== 主程序 ========================
def main():
    # 获取配置名称（从配置文件中读取，可以手动设置）
    config_name = input("请输入本次训练的配置名称（如：test1, H8_pop15等）: ").strip()
    if not config_name:
        config_name = "default"

    # 设置日志
    log_file, timestamp = setup_logging(config_name)

    # 记录配置信息
    log_config()

    logging.info("=" * 60)
    logging.info("BZK-005无人机协同搜索 - 批量测试版")
    logging.info(f"CPU核心数: {cpu_count()}")
    logging.info(f"将依次测试 K = {config.K_LIST}")
    logging.info(f"每个K运行 {config.N_SIM_PER_K} 次仿真")
    logging.info("=" * 60)

    # 设置并行进程数
    if config.N_PROCESSES is None:
        n_processes = min(cpu_count(), 6)
    else:
        n_processes = min(config.N_PROCESSES, cpu_count())

    logging.info(f"\n使用 {n_processes} 个进程并行计算")

    all_results = {}
    total_start_time = time.time()

    for K in config.K_LIST:
        uav_start_K = [(1, 1)] * K
        results = test_single_K(K, config.N_SIM_PER_K, uav_start_K, n_processes)
        all_results[K] = results

        # 实时保存每个K的结果
        if config.SAVE_DETAILED_RESULTS:
            result_file = f'batch_results_K{K}_{config_name}_{timestamp}.json'
        else:
            result_file = f'batch_results_K{K}_{config_name}_{timestamp}_lite.json'

        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        elapsed_total = time.time() - total_start_time
        remaining_K = len(config.K_LIST) - config.K_LIST.index(K) - 1
        avg_time_per_K = elapsed_total / (config.K_LIST.index(K) + 1)
        eta = avg_time_per_K * remaining_K

        logging.info(f"\n总进度: {config.K_LIST.index(K) + 1}/{len(config.K_LIST)}")
        logging.info(f"已用时间: {elapsed_total / 3600:.2f} 小时")
        logging.info(f"预计剩余: {eta / 3600:.2f} 小时")

    # 最终汇总报告
    logging.info("\n" + "=" * 80)
    logging.info("批量测试完成！最终汇总报告")
    logging.info("=" * 80)
    logging.info(f"\n{'K':<5} {'平均时间(h)':<15} {'标准差(h)':<12} {'10h发现率(%)':<12} {'40h发现率(%)':<12}")
    logging.info("-" * 80)

    for K in config.K_LIST:
        res = all_results[K]
        logging.info(f"{K:<5} {res['avg_time_hours']:<15.2f} {res['std_time_hours']:<12.2f} "
                     f"{res['rate_within_10h']:<12.1f} {res['detection_rate']:<12.1f}")

    logging.info("\n满足10小时要求的最小K值分析:")
    for K in config.K_LIST:
        if all_results[K]['rate_within_10h'] >= 95:
            logging.info(f"  ✓ K={K} 可以在10小时内达到 {all_results[K]['rate_within_10h']:.1f}% 的发现率")
            break
    else:
        last_K = config.K_LIST[-1]
        logging.info(f"  ✗ 即使K={last_K}，10小时内发现率仅为 {all_results[last_K]['rate_within_10h']:.1f}%，未达到95%")

    # 保存训练总结
    summary_file = save_training_summary(all_results, config_name, timestamp, log_file)

    # 保存最终报告
    report_file = f'final_report_{config_name}_{timestamp}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("BZK-005无人机协同搜索批量测试报告\n")
        f.write(f"配置名称: {config_name}\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"日志文件: {log_file}\n")
        f.write(f"总结文件: {summary_file}\n\n")
        f.write(f"配置参数:\n")
        f.write(f"  K_LIST = {config.K_LIST}\n")
        f.write(f"  N_SIM_PER_K = {config.N_SIM_PER_K}\n")
        f.write(f"  H = {config.H}\n")
        f.write(f"  pop_size = {config.pop_size}\n")
        f.write(f"  max_iter = {config.max_iter}\n")
        f.write(f"  w_start = {config.w_start}, w_end = {config.w_end}\n")
        f.write(f"  c1 = {config.c1}, c2 = {config.c2}\n")
        f.write(f"  penalty_coef = {config.penalty_coef}\n\n")
        f.write(f"{'K':<5} {'平均时间(h)':<15} {'标准差(h)':<12} {'10h发现率(%)':<12} {'40h发现率(%)':<12}\n")
        f.write("-" * 80 + "\n")

        for K in config.K_LIST:
            res = all_results[K]
            f.write(f"{K:<5} {res['avg_time_hours']:<15.2f} {res['std_time_hours']:<12.2f} "
                    f"{res['rate_within_10h']:<12.1f} {res['detection_rate']:<12.1f}\n")

    logging.info(f"\n所有结果已保存到:")
    logging.info(f"  - 日志文件: {log_file}")
    logging.info(f"  - 训练总结: {summary_file}")
    logging.info(f"  - 最终报告: {report_file}")

    total_time = time.time() - total_start_time
    logging.info(f"\n总耗时: {total_time / 3600:.2f} 小时")


if __name__ == "__main__":
    print("正在初始化Numba编译器...")
    _ = predict_probability_numba(np.ones(M * N), T_mat)
    print("初始化完成！\n")

    main()