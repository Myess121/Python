import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========================= 参数配置 =========================
# 区域与网格
Lx, Ly = 200.0, 200.0  # 区域大小 (km)
R_sensor = 2.0  # 传感器有效探测半径 (km)
dx = dy = 2 * R_sensor  # 网格边长 = 4 km
nx = int(Lx / dx) + 1
ny = int(Ly / dy) + 1

# 目标参数
V_target = 30.0  # 目标最大速度 (km/h)
dt = 5 / 60  # 时间步长 5分钟 = 1/12 h
d_max = int(np.floor(V_target * dt / dx))  # 单步最大移动网格数 (曼哈顿距离)

# 无人机参数
K = 2  # 无人机数量（可调整）
V_u_min, V_u_max = 150, 180  # 速度范围 (km/h)
# 使无人机速度匹配网格移动：每个时间步移动一个网格
V_u = dx / dt  # 需要的速度
if not (V_u_min <= V_u <= V_u_max):
    # 调整dt使速度落在范围内
    V_u = (V_u_min + V_u_max) / 2
    dt = dx / V_u
    d_max = int(np.floor(V_target * dt / dx))
print(f"时间步长 dt = {dt:.3f} h, 无人机速度 = {V_u:.1f} km/h, 目标最大移动网格数 = {d_max}")

# 传感器模型
P_d = 0.95
P_f = 0.05

# 搜索规划参数
lookahead_radius = 10  # 滚动视野半径 (网格数)，限制无人机选择目标点的范围
max_steps = 500  # 单次仿真最大步数
num_simulations = 30  # 蒙特卡洛次数（用于统计平均发现时间）

# 出发点（所有无人机从同一网格出发，假设左下角）
start_grid = (0, 0)


# ========================= 核心函数 =========================
def init_prob_grid():
    """均匀初始化概率图"""
    return np.ones((nx, ny)) / (nx * ny)


def target_motion_transition(prob_grid):
    """马尔可夫转移：曼哈顿距离 <= d_max 均匀传播"""
    new_prob = np.zeros_like(prob_grid)
    # 为每个网格扩散概率到邻域
    for i in range(nx):
        for j in range(ny):
            p = prob_grid[i, j]
            if p == 0:
                continue
            i_min = max(0, i - d_max)
            i_max = min(nx, i + d_max + 1)
            j_min = max(0, j - d_max)
            j_max = min(ny, j + d_max + 1)
            neighbors = []
            for ni in range(i_min, i_max):
                for nj in range(j_min, j_max):
                    if abs(ni - i) + abs(nj - j) <= d_max:
                        neighbors.append((ni, nj))
            if not neighbors:
                continue
            p_share = p / len(neighbors)
            for (ni, nj) in neighbors:
                new_prob[ni, nj] += p_share
    # 归一化
    new_prob /= new_prob.sum()
    return new_prob


def bayesian_update(prob_grid, searched_grids, detected):
    """
    贝叶斯更新
    searched_grids: list of (i,j) 本次搜索的网格
    detected: bool 是否发现目标（本简化中通常为False，因为发现即终止仿真）
    """
    new_prob = prob_grid.copy()
    for (i, j) in searched_grids:
        p_old = prob_grid[i, j]
        if detected:
            p_new = (p_old * P_d) / (p_old * P_d + (1 - p_old) * P_f)
        else:
            p_new = (p_old * (1 - P_d)) / (p_old * (1 - P_d) + (1 - p_old) * (1 - P_f))
        new_prob[i, j] = p_new
    # 归一化（由于数值误差）
    new_prob /= new_prob.sum()
    return new_prob


def choose_search_grids(prob_grid, uav_positions, K, lookahead):
    """
    滚动时域贪心策略：每架无人机在局部视野内选择概率最高的网格
    同时避免多机选择同一网格（简单冲突避免）
    """
    # 计算每个网格的局部概率（只保留无人机附近）
    local_prob = np.zeros_like(prob_grid)
    for idx in range(K):
        pi, pj = uav_positions[idx]
        for di in range(-lookahead, lookahead + 1):
            for dj in range(-lookahead, lookahead + 1):
                ni, nj = pi + di, pj + dj
                if 0 <= ni < nx and 0 <= nj < ny and abs(di) + abs(dj) <= lookahead:
                    local_prob[ni, nj] = prob_grid[ni, nj]  # 用原概率值

    chosen = []
    assigned = set()
    # 按概率降序给无人机分配目标网格
    flat_indices = np.argsort(local_prob, axis=None)[::-1]
    for idx in flat_indices:
        i = idx // ny
        j = idx % ny
        if local_prob[i, j] == 0:
            continue
        if (i, j) not in assigned:
            chosen.append((i, j))
            assigned.add((i, j))
            if len(chosen) == K:
                break
    # 如果不够，用当前位置补齐
    while len(chosen) < K:
        chosen.append(uav_positions[len(chosen)])
    return chosen


def move_towards(pos, target, speed):
    """无人机向目标点移动一步，返回新位置"""
    direction = np.array(target) - np.array(pos)
    dist = np.linalg.norm(direction)
    if dist <= speed:
        return target
    else:
        return pos + (direction / dist) * speed


def simulate_one_run(target_start=None, visualize=False):
    """单次蒙特卡洛仿真，返回发现目标的步数（若未发现返回max_steps）"""
    # 随机初始化目标起始网格
    if target_start is None:
        target_i = np.random.randint(0, nx)
        target_j = np.random.randint(0, ny)
    else:
        target_i, target_j = target_start
    target_pos = (target_i, target_j)

    # 概率图
    prob_grid = init_prob_grid()

    # 无人机初始位置
    uav_positions = [start_grid] * K

    # 轨迹记录（用于可视化）
    trajectories = [[] for _ in range(K)]
    for k in range(K):
        trajectories[k].append(uav_positions[k])

    if visualize:
        # 初始化绘图
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(prob_grid, cmap='hot', origin='lower', vmin=0, vmax=0.01)
        ax.set_title("搜索过程 (按步进)")
        lines = []
        colors = ['cyan', 'lime', 'magenta', 'yellow']
        for k in range(K):
            line, = ax.plot([], [], color=colors[k % len(colors)], linewidth=1.5, label=f'无人机{k + 1}')
            lines.append(line)
        target_point, = ax.plot([], [], 'rx', markersize=10, label='目标真实位置')
        ax.legend()
        plt.pause(0.5)

    # 仿真循环
    for step in range(max_steps):
        # 1. 决策：每架无人机选择目标网格
        target_grids = choose_search_grids(prob_grid, uav_positions, K, lookahead_radius)

        # 2. 移动无人机（假设一步就能飞到目标网格）
        new_uav_pos = []
        for k in range(K):
            new_pos = move_towards(uav_positions[k], target_grids[k], 1.0)  # 速度=1网格/步
            new_uav_pos.append(tuple(map(int, np.round(new_pos))))
        uav_positions = new_uav_pos

        # 3. 探测：检查是否有无人机与目标在同一网格
        detected = False
        for pos in uav_positions:
            if pos == target_pos:
                # 按探测概率决定是否发现
                if np.random.rand() < P_d:
                    detected = True
                    break
        if detected:
            if visualize:
                plt.ioff()
                plt.close()
            return step + 1

        # 4. 贝叶斯更新（本次搜索未发现目标）
        searched = list(set(uav_positions))  # 去重
        prob_grid = bayesian_update(prob_grid, searched, detected=False)

        # 5. 目标运动（随机游走）
        di = np.random.randint(-d_max, d_max + 1)
        dj = np.random.randint(-d_max, d_max + 1)
        while abs(di) + abs(dj) > d_max:
            di = np.random.randint(-d_max, d_max + 1)
            dj = np.random.randint(-d_max, d_max + 1)
        new_i = target_i + di
        new_j = target_j + dj
        new_i = max(0, min(nx - 1, new_i))
        new_j = max(0, min(ny - 1, new_j))
        target_pos = (new_i, new_j)

        # 6. 概率图运动传播
        prob_grid = target_motion_transition(prob_grid)

        # 记录轨迹
        if visualize:
            for k in range(K):
                trajectories[k].append(uav_positions[k])
            # 更新可视化
            if step % 2 == 0:
                ax.clear()
                im = ax.imshow(prob_grid, cmap='hot', origin='lower', vmin=0, vmax=0.01)
                for k, line in enumerate(lines):
                    traj = trajectories[k]
                    if len(traj) > 1:
                        ys, xs = zip(*traj)
                        ax.plot(xs, ys, color=colors[k % len(colors)], linewidth=1.5, label=f'无人机{k + 1}')
                ax.plot(target_pos[1], target_pos[0], 'rx', markersize=10, label='目标')
                ax.set_title(f'Step {step} | 剩余概率 {prob_grid.sum():.3f}')
                ax.legend()
                plt.pause(0.01)

    if visualize:
        plt.ioff()
        plt.close()
    return max_steps


def evaluate_K(K, num_sim=num_simulations):
    """评估给定K下的平均发现步数"""
    steps = []
    for _ in range(num_sim):
        step = simulate_one_run(visualize=False)
        steps.append(step)
    avg_steps = np.mean(steps)
    avg_time = avg_steps * dt
    print(f"K={K}: 平均发现时间 = {avg_time:.2f} 小时 (步数 {avg_steps:.1f})")
    return avg_time


# ========================= 主程序 =========================
if __name__ == "__main__":
    # 确定最少无人机数量（期望10小时内找到）
    T_req = 10.0  # 小时
    K_min = 1
    K_max = 8
    for K in range(K_min, K_max + 1):
        avg_time = evaluate_K(K, num_sim=20)  # 快速测试
        if avg_time <= T_req:
            print(f"满足要求！最少需要 {K} 架无人机")
            # 用更多仿真确认
            avg_time = evaluate_K(K, num_sim=100)
            print(f"详细仿真确认: 平均 {avg_time:.2f} 小时")
            break
    else:
        print(f"K={K_max} 仍未满足要求，请增加上限或调整参数")

    # 演示一次可视化搜索（使用最少数量）
    print("\n开始可视化演示...")
    simulate_one_run(visualize=True)