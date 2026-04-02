import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========================= 参数配置 =========================
# 区域与网格
Lx, Ly = 200.0, 200.0          # 区域大小 (km)
R_sensor = 2.0                 # 传感器有效探测半径 (km)
dx = dy = 2 * R_sensor          # 网格边长 = 4 km
nx = int(Lx / dx) + 1
ny = int(Ly / dy) + 1

# 目标参数
V_target = 30.0                # 目标最大速度 (km/h)
dt = 5 / 60                    # 时间步长 5分钟 = 1/12 h
d_max = int(np.floor(V_target * dt / dx))   # 单步最大移动网格数

# 无人机参数
V_u_min, V_u_max = 150, 180    # 速度范围 (km/h)
V_u = dx / dt
if not (V_u_min <= V_u <= V_u_max):
    V_u = (V_u_min + V_u_max) / 2
    dt = dx / V_u
    d_max = int(np.floor(V_target * dt / dx))
print(f"时间步长 dt = {dt:.3f} h, 无人机速度 = {V_u:.1f} km/h, 目标最大移动网格数 = {d_max}")

# 传感器模型
P_d = 0.95
P_f = 0.05

# 搜索规划参数
lookahead_radius = 10
max_steps = 500
num_sim_fast = 20
num_sim_precise = 80

start_grid = (0, 0)

# ========================= 核心函数 =========================
def init_prob_grid():
    return np.ones((nx, ny)) / (nx * ny)

def target_motion_transition(prob_grid):
    new_prob = np.zeros_like(prob_grid)
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
    new_prob /= new_prob.sum()
    return new_prob

def bayesian_update(prob_grid, searched_grids, detected):
    new_prob = prob_grid.copy()
    for (i, j) in searched_grids:
        p_old = prob_grid[i, j]
        if detected:
            p_new = (p_old * P_d) / (p_old * P_d + (1 - p_old) * P_f)
        else:
            p_new = (p_old * (1 - P_d)) / (p_old * (1 - P_d) + (1 - p_old) * (1 - P_f))
        new_prob[i, j] = p_new
    new_prob /= new_prob.sum()
    return new_prob

def choose_search_grids(prob_grid, uav_positions, K, lookahead):
    local_prob = np.zeros_like(prob_grid)
    for idx in range(K):
        pi, pj = uav_positions[idx]
        for di in range(-lookahead, lookahead+1):
            for dj in range(-lookahead, lookahead+1):
                ni, nj = pi + di, pj + dj
                if 0 <= ni < nx and 0 <= nj < ny and abs(di)+abs(dj) <= lookahead:
                    local_prob[ni, nj] = prob_grid[ni, nj]
    chosen = []
    assigned = set()
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
    while len(chosen) < K:
        chosen.append(uav_positions[len(chosen)])
    return chosen

def move_towards(pos, target, speed):
    direction = np.array(target) - np.array(pos)
    dist = np.linalg.norm(direction)
    if dist <= speed:
        return target
    else:
        return tuple(pos + (direction / dist) * speed)

def simulate_one_run(target_start=None, visualize=False, K=2):
    if target_start is None:
        target_i = np.random.randint(0, nx)
        target_j = np.random.randint(0, ny)
    else:
        target_i, target_j = target_start
    target_pos = (target_i, target_j)

    prob_grid = init_prob_grid()
    uav_positions = [start_grid] * K
    trajectories = [[] for _ in range(K)]
    for k in range(K):
        trajectories[k].append(uav_positions[k])

    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(prob_grid, cmap='hot', origin='lower', vmin=0, vmax=0.01)
        ax.set_title("协同搜索可视化")
        lines = []
        colors = ['cyan', 'lime', 'magenta', 'yellow', 'white']
        for k in range(K):
            line, = ax.plot([], [], color=colors[k%len(colors)], linewidth=1.5, label=f'无人机{k+1}')
            lines.append(line)
        target_point, = ax.plot([], [], 'rx', markersize=10, label='目标')
        ax.legend(loc='upper right')
        plt.pause(0.5)

    for step in range(max_steps):
        target_grids = choose_search_grids(prob_grid, uav_positions, K, lookahead_radius)

        new_uav_pos = []
        for k in range(K):
            new_pos = move_towards(uav_positions[k], target_grids[k], 1.0)
            new_uav_pos.append(tuple(map(int, np.round(new_pos))))
        uav_positions = new_uav_pos

        detected = False
        for pos in uav_positions:
            if pos == target_pos:
                if np.random.rand() < P_d:
                    detected = True
                    break
        if detected:
            if visualize:
                ax.set_title(f"成功！在步数 {step+1} 发现目标")
                fig.canvas.draw()
                plt.ioff()
                plt.show(block=True)   # 等待手动关闭
                plt.close(fig)
            return step + 1

        searched = list(set(uav_positions))
        prob_grid = bayesian_update(prob_grid, searched, detected=False)

        di = np.random.randint(-d_max, d_max+1)
        dj = np.random.randint(-d_max, d_max+1)
        while abs(di) + abs(dj) > d_max:
            di = np.random.randint(-d_max, d_max+1)
            dj = np.random.randint(-d_max, d_max+1)
        new_i = target_i + di
        new_j = target_j + dj
        new_i = max(0, min(nx-1, new_i))
        new_j = max(0, min(ny-1, new_j))
        target_pos = (new_i, new_j)

        prob_grid = target_motion_transition(prob_grid)

        if visualize:
            for k in range(K):
                trajectories[k].append(uav_positions[k])
            if step % 2 == 0:
                ax.clear()
                im = ax.imshow(prob_grid, cmap='hot', origin='lower', vmin=0, vmax=0.01)
                for k, line in enumerate(lines):
                    if k < K:
                        traj = trajectories[k]
                        if len(traj) > 1:
                            ys, xs = zip(*traj)
                            ax.plot(xs, ys, color=colors[k%len(colors)], linewidth=1.5, label=f'无人机{k+1}')
                ax.plot(target_pos[1], target_pos[0], 'rx', markersize=10, label='目标')
                ax.set_title(f'Step {step} | 剩余概率 {prob_grid.sum():.3f}')
                ax.legend(loc='upper right')
                fig.canvas.draw()
                plt.pause(0.01)

    if visualize:
        ax.set_title("超时未发现目标")
        fig.canvas.draw()
        plt.ioff()
        plt.show(block=True)   # 等待手动关闭
        plt.close(fig)
    return max_steps

def evaluate_K(K, num_sim=num_sim_fast):
    steps = []
    for _ in range(num_sim):
        step = simulate_one_run(visualize=False, K=K)
        steps.append(step)
    avg_steps = np.mean(steps)
    avg_time = avg_steps * dt
    print(f"K={K}: 平均发现时间 = {avg_time:.2f} 小时 (步数 {avg_steps:.1f}, 仿真次数={num_sim})")
    return avg_time

if __name__ == "__main__":
    T_req = 10.0
    K_min = 1
    K_max = 8
    best_K = None
    for K_test in range(K_min, K_max+1):
        avg_time = evaluate_K(K_test, num_sim=num_sim_fast)
        if avg_time <= T_req:
            print(f"满足要求！最少需要 {K_test} 架无人机")
            avg_time_precise = evaluate_K(K_test, num_sim=num_sim_precise)
            print(f"精确仿真确认: 平均 {avg_time_precise:.2f} 小时")
            best_K = K_test
            break
    if best_K is None:
        print(f"尝试到{K_max}架仍未满足，将使用{K_max}架进行可视化演示")
        best_K = K_max

    print(f"\n使用 {best_K} 架无人机进行可视化搜索...")
    simulate_one_run(visualize=True, K=best_K)
    print("可视化窗口已关闭。")