import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========================= 参数配置 =========================
Lx, Ly = 200.0, 200.0  # 区域大小 (km)
R_sensor = 2.0  # 传感器探测半径 (km)
dx = dy = 2 * R_sensor  # 网格边长 4km
nx = int(Lx / dx) + 1
ny = int(Ly / dy) + 1

V_target = 30.0  # 目标最大速度 km/h
dt = 5 / 60  # 时间步长 h
d_max = int(np.floor(V_target * dt / dx))

# 无人机参数
V_u_min, V_u_max = 150, 180
V_u = dx / dt
if not (V_u_min <= V_u <= V_u_max):
    V_u = (V_u_min + V_u_max) / 2
    dt = dx / V_u
    d_max = int(np.floor(V_target * dt / dx))
print(f"dt={dt:.3f}h, 无人机速度={V_u:.1f}km/h, 目标最大移动网格数={d_max}")

# 传感器
P_d, P_f = 0.95, 0.05

# 搜索规划
lookahead_radius = 10
max_steps = 500
num_sim_fast = 20
num_sim_precise = 80
start_grid = (0, 0)

# 新增：无人机续航（单位：步数），假设续航40小时
endurance_steps = int(40 / dt)  # 40小时对应的步数
print(f"无人机续航步数: {endurance_steps}")

# 新增：最小安全距离（网格数）
safe_distance = 2  # 两架无人机之间至少相隔2个网格


# ========================= 核心函数 =========================
def init_prob_grid():
    return np.ones((nx, ny)) / (nx * ny)


def target_motion_transition(prob_grid):
    new_prob = np.zeros_like(prob_grid)
    for i in range(nx):
        for j in range(ny):
            p = prob_grid[i, j]
            if p == 0: continue
            i_min, i_max = max(0, i - d_max), min(nx, i + d_max + 1)
            j_min, j_max = max(0, j - d_max), min(ny, j + d_max + 1)
            neighbors = [(ni, nj) for ni in range(i_min, i_max) for nj in range(j_min, j_max)
                         if abs(ni - i) + abs(nj - j) <= d_max]
            if not neighbors: continue
            for ni, nj in neighbors:
                new_prob[ni, nj] += p / len(neighbors)
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


def region_partition(K, uav_index):
    """
    区域分割：将网格按列（x方向）分成K个条带，返回当前无人机应优先搜索的列范围
    返回 (min_col, max_col) 闭区间
    """
    col_width = ny // K
    start_col = uav_index * col_width
    end_col = ny - 1 if uav_index == K - 1 else start_col + col_width - 1
    return start_col, end_col


def choose_search_grids_region(prob_grid, uav_positions, K, lookahead, step_count):
    """
    改进的决策函数：区域分割 + 视野内贪心 + 避让（避免多机选同一网格）
    返回每架无人机的目标网格 (i, j)
    """
    targets = []
    # 1. 每架无人机在各自区域内找局部最高概率网格
    for k in range(K):
        # 如果无人机已耗尽续航，不再规划移动
        if step_count[k] >= endurance_steps:
            targets.append(uav_positions[k])  # 原地不动
            continue

        col_min, col_max = region_partition(K, k)
        # 在视野内且属于本区域的网格中找最高概率
        pi, pj = uav_positions[k]
        best_prob = -1
        best_pos = uav_positions[k]  # 默认当前位置
        for di in range(-lookahead, lookahead + 1):
            for dj in range(-lookahead, lookahead + 1):
                if abs(di) + abs(dj) > lookahead: continue
                ni, nj = pi + di, pj + dj
                if 0 <= ni < nx and 0 <= nj < ny and col_min <= nj <= col_max:
                    if prob_grid[ni, nj] > best_prob:
                        best_prob = prob_grid[ni, nj]
                        best_pos = (ni, nj)
        # 如果区域内视野无有效网格，则扩大到全图视野（但不跨区？可放宽）
        if best_prob < 1e-9:
            # 全图视野内找最高概率（仍优先区域？简化：全图）
            for di in range(-lookahead, lookahead + 1):
                for dj in range(-lookahead, lookahead + 1):
                    if abs(di) + abs(dj) > lookahead: continue
                    ni, nj = pi + di, pj + dj
                    if 0 <= ni < nx and 0 <= nj < ny:
                        if prob_grid[ni, nj] > best_prob:
                            best_prob = prob_grid[ni, nj]
                            best_pos = (ni, nj)
        targets.append(best_pos)

    # 2. 冲突消解：如果有两架无人机目标网格距离小于安全距离，调整优先级较低的无人机
    # 按概率值排序，高概率的优先保留
    # 先计算每个目标网格的概率值
    target_probs = [prob_grid[ti, tj] for (ti, tj) in targets]
    order = np.argsort(target_probs)[::-1]  # 概率高的在前
    assigned = set()
    final_targets = [None] * K
    for idx in order:
        pos = targets[idx]
        # 检查是否与已分配的位置冲突（距离<safe_distance）
        conflict = False
        for other_idx in assigned:
            other_pos = final_targets[other_idx]
            if abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1]) < safe_distance:
                conflict = True
                break
        if not conflict:
            final_targets[idx] = pos
            assigned.add(idx)
        else:
            # 冲突：在视野内找次优且不冲突的位置
            pi, pj = uav_positions[idx]
            # 搜索视野内所有网格，按概率降序，选第一个不冲突的
            candidates = []
            for di in range(-lookahead, lookahead + 1):
                for dj in range(-lookahead, lookahead + 1):
                    if abs(di) + abs(dj) > lookahead: continue
                    ni, nj = pi + di, pj + dj
                    if 0 <= ni < nx and 0 <= nj < ny:
                        candidates.append((ni, nj, prob_grid[ni, nj]))
            candidates.sort(key=lambda x: x[2], reverse=True)
            chosen = None
            for (ni, nj, _) in candidates:
                ok = True
                for other_idx in assigned:
                    other_pos = final_targets[other_idx]
                    if abs(ni - other_pos[0]) + abs(nj - other_pos[1]) < safe_distance:
                        ok = False
                        break
                if ok:
                    chosen = (ni, nj)
                    break
            if chosen is None:
                chosen = uav_positions[idx]  # 实在不行就原地
            final_targets[idx] = chosen
            assigned.add(idx)
    return final_targets


def move_towards(pos, target):
    """一步移动一个网格（方向取整）"""
    dr = target[0] - pos[0]
    dc = target[1] - pos[1]
    if dr == 0 and dc == 0:
        return pos
    # 优先移动曼哈顿距离较大的方向
    if abs(dr) >= abs(dc):
        step_r = 1 if dr > 0 else (-1 if dr < 0 else 0)
        step_c = 0
    else:
        step_r = 0
        step_c = 1 if dc > 0 else (-1 if dc < 0 else 0)
    return (pos[0] + step_r, pos[1] + step_c)


def simulate_one_run(target_start=None, visualize=False, K=2):
    if target_start is None:
        target_i, target_j = np.random.randint(0, nx), np.random.randint(0, ny)
    else:
        target_i, target_j = target_start
    target_pos = (target_i, target_j)

    prob_grid = init_prob_grid()
    uav_positions = [start_grid] * K
    # 记录每架无人机已飞行步数
    step_count = [0] * K
    trajectories = [[] for _ in range(K)]
    for k in range(K):
        trajectories[k].append(uav_positions[k])

    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(prob_grid, cmap='hot', origin='lower', vmin=0, vmax=0.01)
        ax.set_title("协同搜索 (区域分割+避碰+续航)")
        lines = []
        colors = ['cyan', 'lime', 'magenta', 'yellow', 'white']
        for k in range(K):
            line, = ax.plot([], [], color=colors[k % len(colors)], linewidth=1.5, label=f'UAV{k + 1}')
            lines.append(line)
        target_point, = ax.plot([], [], 'rx', markersize=10, label='目标')
        ax.legend(loc='upper right')
        plt.pause(0.5)

    for step in range(max_steps):
        # 1. 决策目标网格（考虑区域分割和避碰）
        targets = choose_search_grids_region(prob_grid, uav_positions, K, lookahead_radius, step_count)

        # 2. 移动无人机（一步一网格）
        new_positions = []
        for k in range(K):
            if step_count[k] >= endurance_steps:
                new_positions.append(uav_positions[k])  # 续航耗尽，原地不动
            else:
                new_pos = move_towards(uav_positions[k], targets[k])
                new_positions.append(new_pos)
                step_count[k] += 1
        uav_positions = new_positions

        # 3. 探测
        detected = False
        for pos in uav_positions:
            if pos == target_pos and np.random.rand() < P_d:
                detected = True
                break
        if detected:
            if visualize:
                ax.set_title(f"成功！步数 {step + 1}")
                fig.canvas.draw()
                plt.ioff()
                plt.show(block=True)
                plt.close(fig)
            return step + 1

        # 4. 贝叶斯更新（未发现）
        searched = list(set(uav_positions))
        prob_grid = bayesian_update(prob_grid, searched, detected=False)

        # 5. 目标运动
        di = np.random.randint(-d_max, d_max + 1)
        dj = np.random.randint(-d_max, d_max + 1)
        while abs(di) + abs(dj) > d_max:
            di = np.random.randint(-d_max, d_max + 1)
            dj = np.random.randint(-d_max, d_max + 1)
        new_i = max(0, min(nx - 1, target_i + di))
        new_j = max(0, min(ny - 1, target_j + dj))
        target_pos = (new_i, new_j)
        target_i, target_j = target_pos

        # 6. 概率图传播
        prob_grid = target_motion_transition(prob_grid)

        # 记录轨迹和可视化
        for k in range(K):
            trajectories[k].append(uav_positions[k])
        if visualize and step % 2 == 0:
            ax.clear()
            im = ax.imshow(prob_grid, cmap='hot', origin='lower', vmin=0, vmax=0.01)
            for k in range(K):
                if k < len(trajectories) and len(trajectories[k]) > 1:
                    ys, xs = zip(*trajectories[k])
                    ax.plot(xs, ys, color=colors[k % len(colors)], linewidth=1.5, label=f'UAV{k + 1}')
            ax.plot(target_pos[1], target_pos[0], 'rx', markersize=10, label='目标')
            # 显示剩余续航
            remaining = [max(0, endurance_steps - step_count[k]) for k in range(K)]
            title = f'Step {step} | 剩余概率 {prob_grid.sum():.3f} | 续航剩余: {remaining}'
            ax.set_title(title)
            ax.legend(loc='upper right')
            fig.canvas.draw()
            plt.pause(0.01)

    if visualize:
        ax.set_title("超时未发现")
        fig.canvas.draw()
        plt.ioff()
        plt.show(block=True)
        plt.close(fig)
    return max_steps


def evaluate_K(K, num_sim=num_sim_fast):
    steps = []
    for _ in range(num_sim):
        step = simulate_one_run(visualize=False, K=K)
        steps.append(step)
    avg_steps = np.mean(steps)
    avg_time = avg_steps * dt
    print(f"K={K}: 平均发现时间={avg_time:.2f}h (步数{avg_steps:.1f}, 仿真{num_sim}次)")
    return avg_time


if __name__ == "__main__":
    T_req = 10.0
    K_max = 8
    best_K = None
    for K_test in range(1, K_max + 1):
        avg_t = evaluate_K(K_test, num_sim=num_sim_fast)
        if avg_t <= T_req:
            print(f"满足要求！最少需要 {K_test} 架无人机")
            avg_t2 = evaluate_K(K_test, num_sim=num_sim_precise)
            print(f"精确确认: {avg_t2:.2f}h")
            best_K = K_test
            break
    if best_K is None:
        best_K = K_max
        print(f"尝试到{K_max}架未满足，使用{K_max}架演示")

    print(f"\n使用 {best_K} 架无人机可视化演示...")
    simulate_one_run(visualize=True, K=best_K)
    print("可视化窗口已关闭。")