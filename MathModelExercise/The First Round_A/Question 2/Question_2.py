# -*- coding: utf-8 -*-
"""
整合版：BZK-005无人机协同搜索
用户可输入无人机数量K和蒙特卡洛仿真次数N，
先进行N次无动画仿真统计平均发现时间，
最后进行一次带动画的仿真展示搜索过程。
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ======================== 参数设置 ========================
M, N = 16, 23                     # 网格数 (纬度, 经度)
dx, dy = 20.0, 20.0               # 网格边长 (km)
V_uav = 180.0                     # 无人机速度 (km/h)
dt = dx / V_uav                   # 时间步长 (h)
max_flight_time = 40.0            # 最大续航 (h)
max_steps = int(max_flight_time / dt)   # 约360步
start_step = 23                   # 第23步开始探测（前22步入场）

Pd = 0.9                          # 探测概率
Pf = 0.05                         # 虚警概率

# 目标运动参数
p_stay = 0.95                     # 留在原网格概率
p_move = 0.05                     # 移动到邻域总概率

# 滚动时域参数
H = 6                             # 规划步数（40分钟）

# CC-MPSO 参数（可调低以加快速度）
pop_size = 15                     # 粒子数
max_iter = 30                     # 迭代次数
coop_gap = 5
w_start, w_end = 0.9, 0.4
c1, c2 = 1.5, 1.5
penalty_coef = 1000.0

# ======================== 辅助函数 ========================
def get_neighbors(i, j):
    neighbors = []
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni, nj = i+di, j+dj
        if 1 <= ni <= M and 1 <= nj <= N:
            neighbors.append((ni, nj))
    return neighbors

def build_transition_matrix():
    n_cells = M * N
    T = np.zeros((n_cells, n_cells))
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

T_mat = build_transition_matrix()

def predict_probability(p_curr):
    p_flat = p_curr.flatten()
    new_flat = T_mat @ p_flat
    return new_flat.reshape((M, N))

def bayesian_update(p_grid, detected, i, j):
    prior = p_grid[i-1, j-1]
    if detected:
        post = (prior * Pd) / (prior * Pd + (1-prior) * Pf)
    else:
        post = (prior * (1-Pd)) / (prior * (1-Pd) + (1-prior) * (1-Pf))
    p_grid[i-1, j-1] = post
    p_grid /= p_grid.sum()
    return p_grid

def generate_observation(true_pos, probe_pos):
    if true_pos == probe_pos:
        return random.random() < Pd
    else:
        return random.random() < Pf

def move_target(pos):
    i, j = pos
    if random.random() < p_stay:
        return (i, j)
    neighbors = get_neighbors(i, j)
    if neighbors:
        return random.choice(neighbors)
    return (i, j)

# ======================== CC-MPSO 类（支持外部解码函数） ========================
class CC_MPSO:
    def __init__(self, K, H, pop_size, max_iter, coop_gap, w_start, w_end, c1, c2, fitness_func):
        self.K = K
        self.H = H
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.coop_gap = coop_gap
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.fitness_func = fitness_func
        self.decode_func = None   # 外部解码函数
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

    def decode_trajectory(self, dirs, start_pos):
        if self.decode_func is not None:
            return self.decode_func(dirs, start_pos)
        # 默认解码（边界限制）
        traj = [start_pos]
        ci, cj = start_pos
        for d in dirs:
            if d == 0: ci = max(1, ci-1)
            elif d == 1: ci = min(M, ci+1)
            elif d == 2: cj = max(1, cj-1)
            elif d == 3: cj = min(N, cj+1)
            traj.append((ci, cj))
        return traj

    def evaluate(self, all_dirs, start_positions, p_curr):
        p_seq = [p_curr]
        for _ in range(self.H):
            p_seq.append(predict_probability(p_seq[-1]))
        total_reward = 0.0
        penalty = 0.0
        for h in range(self.H):
            positions = []
            for k in range(self.K):
                traj = self.decode_trajectory(all_dirs[k], start_positions[k])
                pos = traj[h+1]
                positions.append(pos)
            if len(set(positions)) != len(positions):
                penalty += penalty_coef
            for k, pos in enumerate(positions):
                i, j = pos
                total_reward += p_seq[h+1][i-1, j-1]
        return total_reward * Pd - penalty

    def optimize(self, start_positions, p_curr):
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
                fit = self.evaluate(all_dirs, start_positions, p_curr)
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
                    new_p = np.round(new_p).astype(int)
                    sub['particles'][i] = new_p
                for i in range(self.pop_size):
                    dirs = sub['particles'][i]
                    all_dirs = []
                    for k in range(self.K):
                        if k == sidx:
                            all_dirs.append(dirs)
                        else:
                            all_dirs.append(self.subpop[k]['gbest'])
                    fit = self.evaluate(all_dirs, start_positions, p_curr)
                    if fit > sub['pbest_fit'][i]:
                        sub['pbest_fit'][i] = fit
                        sub['pbest'][i] = dirs.copy()
                    if fit > sub['gbest_fit']:
                        sub['gbest_fit'] = fit
                        sub['gbest'] = dirs.copy()
        return [sub['gbest'] for sub in self.subpop]

# ======================== 无动画仿真（用于统计） ========================
def run_simulation_no_animation(K, target_init_pos, uav_start_positions):
    """单次仿真，不绘图，返回发现时间步数（若未发现返回None）"""
    p = np.ones((M, N)) / (M * N)
    target_pos = target_init_pos
    uav_pos = list(uav_start_positions)
    total_steps_done = start_step - 1
    detected = False
    discovery_step = None

    def decode_dirs(dirs, start):
        traj = [start]
        ci, cj = start
        for d in dirs:
            if d == 0: ci = max(1, ci-1)
            elif d == 1: ci = min(M, ci+1)
            elif d == 2: cj = max(1, cj-1)
            elif d == 3: cj = min(N, cj+1)
            traj.append((ci, cj))
        return traj

    while total_steps_done < max_steps and not detected:
        remaining = max_steps - total_steps_done
        horizon = min(H, remaining)
        if horizon <= 0:
            break

        def fitness_func(all_dirs):
            p_seq = [p]
            for _ in range(horizon):
                p_seq.append(predict_probability(p_seq[-1]))
            total = 0.0
            penalty = 0.0
            for h in range(horizon):
                positions = []
                for k in range(K):
                    traj = decode_dirs(all_dirs[k], uav_pos[k])
                    pos = traj[h+1]
                    positions.append(pos)
                if len(set(positions)) != len(positions):
                    penalty += penalty_coef
                for k, pos in enumerate(positions):
                    i, j = pos
                    total += p_seq[h+1][i-1, j-1]
            return total * Pd - penalty

        optimizer = CC_MPSO(K, horizon, pop_size, max_iter, coop_gap,
                            w_start, w_end, c1, c2, fitness_func)
        optimizer.decode_func = decode_dirs
        best_dirs = optimizer.optimize(uav_pos, p)

        # 执行整个规划周期
        for step in range(horizon):
            if total_steps_done >= max_steps or detected:
                break
            new_positions = []
            for k in range(K):
                d = best_dirs[k][step]
                ci, cj = uav_pos[k]
                if d == 0: ci = max(1, ci-1)
                elif d == 1: ci = min(M, ci+1)
                elif d == 2: cj = max(1, cj-1)
                elif d == 3: cj = min(N, cj+1)
                new_positions.append((ci, cj))
            # 探测更新
            for k, pos in enumerate(new_positions):
                obs = generate_observation(target_pos, pos)
                p = bayesian_update(p, obs, pos[0], pos[1])
                if obs and pos == target_pos:
                    detected = True
                    discovery_step = total_steps_done + 1
                    break
            if detected:
                break
            uav_pos = new_positions
            total_steps_done += 1
            target_pos = move_target(target_pos)
            p = predict_probability(p)
    return discovery_step

# ======================== 带动画仿真 ========================
def run_simulation_with_animation(K, uav_start_positions):
    """运行一次带动画的仿真，返回发现时间步数"""
    p = np.ones((M, N)) / (M * N)
    target_pos = (random.randint(1, M), random.randint(1, N))
    uav_pos = list(uav_start_positions)
    total_steps_done = start_step - 1
    detected = False
    discovery_step = None

    history_uav = [[] for _ in range(K)]
    for k in range(K):
        history_uav[k].append(uav_pos[k])
    history_target = [target_pos]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(p, cmap='hot', interpolation='bilinear', origin='upper', vmin=0, vmax=0.02)
    ax.set_title(f"Target Probability Map (K={K})")
    ax.set_xlabel("Longitude grid (j)")
    ax.set_ylabel("Latitude grid (i)")
    colors = ['bo', 'go', 'ro', 'co', 'mo', 'yo']
    markers = []
    trails = []
    for k in range(K):
        m, = ax.plot([], [], colors[k%len(colors)], markersize=10, label=f'UAV {k+1}')
        t, = ax.plot([], [], '--', linewidth=1, alpha=0.5, color=colors[k%len(colors)][0])
        markers.append(m)
        trails.append(t)
    target_marker, = ax.plot([], [], 'r*', markersize=12, label='Target')
    ax.legend()

    def decode_dirs(dirs, start):
        traj = [start]
        ci, cj = start
        for d in dirs:
            if d == 0: ci = max(1, ci-1)
            elif d == 1: ci = min(M, ci+1)
            elif d == 2: cj = max(1, cj-1)
            elif d == 3: cj = min(N, cj+1)
            traj.append((ci, cj))
        return traj

    def update(frame):
        nonlocal p, uav_pos, target_pos, total_steps_done, detected, discovery_step
        if detected or total_steps_done >= max_steps:
            anim.event_source.stop()
            return [im] + markers + [target_marker] + trails

        remaining = max_steps - total_steps_done
        horizon = min(H, remaining)
        if horizon <= 0:
            anim.event_source.stop()
            return [im] + markers + [target_marker] + trails

        def fitness_func(all_dirs):
            p_seq = [p]
            for _ in range(horizon):
                p_seq.append(predict_probability(p_seq[-1]))
            total = 0.0
            penalty = 0.0
            for h in range(horizon):
                positions = []
                for k in range(K):
                    traj = decode_dirs(all_dirs[k], uav_pos[k])
                    pos = traj[h+1]
                    positions.append(pos)
                if len(set(positions)) != len(positions):
                    penalty += penalty_coef
                for k, pos in enumerate(positions):
                    i, j = pos
                    total += p_seq[h+1][i-1, j-1]
            return total * Pd - penalty

        optimizer = CC_MPSO(K, horizon, pop_size, max_iter, coop_gap,
                            w_start, w_end, c1, c2, fitness_func)
        optimizer.decode_func = decode_dirs
        best_dirs = optimizer.optimize(uav_pos, p)

        # 只执行一步（动画）
        step = 0
        new_positions = []
        for k in range(K):
            d = best_dirs[k][step]
            ci, cj = uav_pos[k]
            if d == 0: ci = max(1, ci-1)
            elif d == 1: ci = min(M, ci+1)
            elif d == 2: cj = max(1, cj-1)
            elif d == 3: cj = min(N, cj+1)
            new_positions.append((ci, cj))

        for k, pos in enumerate(new_positions):
            obs = generate_observation(target_pos, pos)
            p = bayesian_update(p, obs, pos[0], pos[1])
            if obs and pos == target_pos and not detected:
                detected = True
                discovery_step = total_steps_done + 1
                print(f"\n>>> 动画中发现目标！步数: {discovery_step}, 时间: {discovery_step*dt:.2f} h")

        uav_pos = new_positions
        total_steps_done += 1
        target_pos = move_target(target_pos)
        p = predict_probability(p)

        for k in range(K):
            history_uav[k].append(uav_pos[k])
        history_target.append(target_pos)

        im.set_data(p)
        for k in range(K):
            markers[k].set_data([history_uav[k][-1][1]-1], [history_uav[k][-1][0]-1])
            if len(history_uav[k]) > 1:
                trails[k].set_data([p[1]-1 for p in history_uav[k]], [p[0]-1 for p in history_uav[k]])
        target_marker.set_data([history_target[-1][1]-1], [history_target[-1][0]-1])
        ax.set_title(f"Step {total_steps_done} (time {total_steps_done*dt:.1f}h) | Detected: {detected}")

        if detected:
            anim.event_source.stop()
        return [im] + markers + [target_marker] + trails

    anim = FuncAnimation(fig, update, frames=max_steps, interval=50, repeat=False)
    plt.show()
    return discovery_step

# ======================== 主程序 ========================
def main():
    print("="*60)
    print("BZK-005无人机协同搜索 - 整合版")
    print("适用问题一（求最少无人机数满足10小时）和问题二（2架无人机的期望时间）")
    print("="*60)
    try:
        K = int(input("请输入无人机数量 K (例如 2): "))
        N_sim = int(input("请输入蒙特卡洛仿真次数 (建议 5~20): "))
    except:
        K = 2
        N_sim = 5
        print(f"输入无效，使用默认值 K={K}, N_sim={N_sim}")

    uav_start = [(1,1)] * K   # 所有无人机从左上角出发

    # 无动画统计
    print(f"\n正在运行 {N_sim} 次无动画仿真 (K={K})...")
    detect_steps = []
    for i in range(N_sim):
        init_pos = (random.randint(1, M), random.randint(1, N))
        step = run_simulation_no_animation(K, init_pos, uav_start)
        if step is not None:
            detect_steps.append(step)
        else:
            detect_steps.append(max_steps)
        print(f"  进度 {i+1}/{N_sim}, 发现步数: {step if step else '未发现'}")
    avg_step = np.mean(detect_steps)
    std_step = np.std(detect_steps)
    avg_time = avg_step * dt
    std_time = std_step * dt
    print(f"\n统计结果 (K={K}, {N_sim}次):")
    print(f"  平均发现步数: {avg_step:.1f} 步")
    print(f"  平均发现时间: {avg_time:.2f} ± {std_time:.2f} 小时")

    # 询问是否展示动画
    show_anim = input("\n是否运行一次带动画的仿真来展示搜索过程？(y/n): ").strip().lower()
    if show_anim == 'y':
        print("正在启动动画仿真...")
        anim_step = run_simulation_with_animation(K, uav_start)
        if anim_step:
            print(f"动画仿真发现时间: {anim_step * dt:.2f} 小时")
        else:
            print("动画仿真未发现目标")

if __name__ == "__main__":
    main()