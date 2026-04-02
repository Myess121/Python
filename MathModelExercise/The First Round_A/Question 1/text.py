import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# --- 修复 1: 解决中文乱码 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 核心参数设置 (根据你的截图优化) =================
GRID_SIZE = 60
SENSOR_RADIUS = 5  # 略微增大探测半径，增强视觉效果
LOOKAHEAD_RADIUS = 20
DRONE_SPEED = 2.5
TARGET_SPEED = 0.5  # 目标机动扩散速度 (马尔可夫演化)
STEPS = 500

# ================= 2. 初始化概率图  =================
# 初始概率均匀分布
prob_map = np.ones((GRID_SIZE, GRID_SIZE)) / (GRID_SIZE ** 2)
drone_pos = np.array([5.0, 5.0])
trajectory = [drone_pos.copy()]

plt.ion()
fig, ax = plt.subplots(figsize=(8, 7))

# ================= 3. 动态寻找主循环 =================
for step in range(STEPS):
    # 【机制1：马尔可夫概率传播】模拟目标位置随时间的演化 [cite: 7, 19]
    prob_map = gaussian_filter(prob_map, sigma=TARGET_SPEED)

    # 【机制2：RHC局部寻优】寻找视距内的最高概率点 [cite: 46, 50]
    Y, X = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    dist_sq_to_drone = (Y - drone_pos[0]) ** 2 + (X - drone_pos[1]) ** 2
    local_mask = dist_sq_to_drone <= LOOKAHEAD_RADIUS ** 2

    if np.sum(prob_map[local_mask]) > 1e-7:
        local_prob = np.copy(prob_map)
        local_prob[~local_mask] = 0
        target_idx = np.unravel_index(np.argmax(local_prob), local_prob.shape)
    else:
        target_idx = np.unravel_index(np.argmax(prob_map), prob_map.shape)

    target_pos = np.array(target_idx)

    # 【机制3：机动约束】朝目标点匀速飞行 [cite: 54, 62]
    direction = target_pos - drone_pos
    dist = np.linalg.norm(direction)
    if dist > 0:
        drone_pos += (direction / dist) * min(DRONE_SPEED, dist)

    trajectory.append(drone_pos.copy())

    # 【机制4：贝叶斯观测更新】未发现目标，网格概率降为0 [cite: 13, 14]
    sensor_mask = ((Y - drone_pos[0]) ** 2 + (X - drone_pos[1]) ** 2) <= SENSOR_RADIUS ** 2
    prob_map[sensor_mask] = 0

    # ================= 4. 绘图渲染 (优化对比度) =================
    if step % 2 == 0:
        ax.clear()
        # 调整 vmax，让微小的概率变化也能呈现色彩差异
        im = ax.imshow(prob_map, cmap='hot', origin='lower', vmin=0, vmax=np.max(prob_map) * 0.8)

        traj_y, traj_x = zip(*trajectory)
        ax.plot(traj_x, traj_y, color='cyan', linewidth=1, alpha=0.6, label='飞行航迹')
        ax.plot(drone_pos[1], drone_pos[0], 'bo', markersize=6, label='无人机')
        ax.plot(target_pos[1], target_pos[0], 'rx', markersize=8, label='RHC目标点')

        remaining_prob = np.sum(prob_map)
        ax.set_title(f"步数: {step} | 剩余未发现概率: {remaining_prob:.2%}")
        ax.legend(loc='upper right')
        plt.pause(0.01)

plt.ioff()
plt.show()