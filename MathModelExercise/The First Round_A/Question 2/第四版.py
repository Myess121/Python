import numpy as np
import random
import sys

# ================= 严格对齐文档的参数 =================
M, N = 16, 23
GRID_SIZE = 20  # km
PD = 0.9  # 探测概率
PF = 0.05  # 虚警概率
DT = 1 / 9  # 步长 (6.67min)
TRANSITION_STAY = 0.95  # 留在原网格概率
# 抵达边界所需步数：450km / 180km/h = 2.5h = 22.5步，取23步
TRANSIT_STEPS = 23


def run_single_simulation(num_uavs):
    # 1. 目标初始化
    target_pos = [random.randint(0, M - 1), random.randint(0, N - 1)]

    # 2. 概率图初始化 (均匀分布)
    prob_map = np.full((M, N), 1.0 / (M * N))

    # 3. 无人机初始化：第23步到达(0,0), (0,1)等入口点
    uav_positions = []
    for k in range(num_uavs):
        uav_positions.append([0, min(k, N - 1)])

        # 初始步数直接计入从机场飞来的时间
    total_steps = TRANSIT_STEPS
    found = False

    # 记录每个网格被搜索过的次数，用于简单的多机协同（避开已搜区域）
    search_history = np.zeros((M, N))

    while total_steps < 360:  # 最大40小时
        total_steps += 1

        # --- A. 探测与贝叶斯更新 ---
        for k in range(num_uavs):
            curr_pos = tuple(uav_positions[k])

            # 检查是否发现目标
            if curr_pos == (target_pos[0], target_pos[1]) and random.random() < PD:
                return total_steps

            # 更新概率图：对已探测网格进行后验更新
            p_ij = prob_map[curr_pos]
            prob_map[curr_pos] = (p_ij * (1 - PD)) / (p_ij * (1 - PD) + (1 - p_ij) * (1 - PF))
            search_history[curr_pos] += 1  # 标记此地已被搜过

        # 归一化
        prob_map /= prob_map.sum()

        # --- B. 目标机动 (马尔可夫) ---
        if random.random() > TRANSITION_STAY:
            move = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            nt_r, nt_c = target_pos[0] + move[0], target_pos[1] + move[1]
            if 0 <= nt_r < M and 0 <= nt_c < N:
                target_pos = [nt_r, nt_c]

        # --- C. 协同决策：滚动时域简化版 ---
        # 核心逻辑：计算吸引力 = 概率图 / (1 + 搜索次数)，强迫无人机分散
        for k in range(num_uavs):
            r, c = uav_positions[k]
            best_score = -1
            next_move = [r, c]

            # 候选动作：上下左右及原地
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < M and 0 <= nc < N:
                    # 避碰约束：不同无人机不能在同一网格
                    if any([[nr, nc] == uav_positions[i] for i in range(num_uavs) if i != k]):
                        continue

                    # 协同收益函数：概率越高越好，被搜过次数越多越差
                    score = prob_map[nr, nc] / (search_history[nr, nc] + 1)
                    if score > best_score:
                        best_score = score
                        next_move = [nr, nc]
            uav_positions[k] = next_move

    return 360


def main():
    try:
        num_uavs = int(input("请输入无人机数量: "))
    except:
        return

    results = []
    moving_avg = 0
    stable_window = []

    print(f"正在启动 BZK-005 协同搜索仿真 (含 {TRANSIT_STEPS} 步进场时间)...")

    for i in range(1, 1001):  # 最高1000次
        step = run_single_simulation(num_uavs)
        results.append(step)

        current_avg = np.mean(results)

        if i % 20 == 0:
            print(f"次数: {i} | 当前平均发现时间: {current_avg * DT:.2f} 小时")

        # 稳定性判定
        stable_window.append(current_avg)
        if len(stable_window) > 30:
            stable_window.pop(0)
            std_dev = np.std(stable_window)
            # 如果最近30次波动的标准差极小，则停止
            if i > 60 and std_dev < 0.05:
                print(f"结果已收敛。")
                break

    print("\n" + "=" * 40)
    print(f"无人机数量: {num_uavs}")
    print(f"最终仿真次数: {i}")
    print(f"平均发现时间: {np.mean(results) * DT:.2f} 小时")
    print("=" * 40)


if __name__ == "__main__":
    main()