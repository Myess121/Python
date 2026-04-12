# run_swarm_collaboration.py
"""
多无人机协同仿真脚本
验证：任务分配 + 通信感知独立规划 + 防碰撞 + 集群指标统计
"""
import os
import csv
import math
import matplotlib.pyplot as plt

from config import *
from environment import GridMap, GroundStation, Jammer
from swarm.uav import UAV
from swarm.swarm_planner import SwarmPlanner


def calculate_inter_uav_distances(uavs, step):
    """计算同一时间步所有 UAV 的两两距离，返回最小值（防碰撞指标）"""
    dists = []
    positions = []
    for u in uavs:
        if step < len(u.path):
            positions.append(u.path[step])
        else:
            positions.append(u.path[-1])

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = math.hypot(positions[i][0] - positions[j][0],
                           positions[i][1] - positions[j][1])
            dists.append(d)
    return min(dists) if dists else float('inf')


def main():
    print("🚀 开始多无人机协同仿真...")
    os.makedirs("results", exist_ok=True)

    # 1. 初始化环境
    grid_map = GridMap()
    gs = GroundStation()
    jammers = [Jammer(pos=(25.0, 15.0), velocity=(0.5, 0.3))]

    # 2. 定义 3 架无人机（侦察+中继角色）
    uavs = [
        UAV(0, start_pos=(5, 5), goal_pos=(45, 25), role="scout"),
        UAV(1, start_pos=(5, 15), goal_pos=(45, 15), role="relay"),
        UAV(2, start_pos=(5, 25), goal_pos=(45, 5), role="scout")
    ]

    # 3. 执行协同规划（内部自动调用 task_assignment）
    planner = SwarmPlanner(uavs, grid_map, gs, jammers)
    all_paths = planner.plan_collaboratively()

    # 将规划结果绑定回 UAV 实例
    for uav in uavs:
        uav.path = all_paths[uav.id]

    # 4. 计算集群指标
    max_path_len = max(len(u.path) for u in uavs)
    min_inter_dist = float('inf')

    for t in range(max_path_len):
        d = calculate_inter_uav_distances(uavs, t)
        if d < min_inter_dist:
            min_inter_dist = d

    # 5. 可视化
    plt.figure(figsize=(10, 6))
    if grid_map.obstacles:
        obs_x = [p[0] for p in grid_map.obstacles]
        obs_y = [p[1] for p in grid_map.obstacles]
        plt.plot(obs_x, obs_y, 'k^', markersize=8, label='Obstacles')

    plt.plot(jammers[0].pos[0], jammers[0].pos[1], 'r*', markersize=20, label='Jammer')

    colors = ['blue', 'green', 'orange']
    for i, uav in enumerate(uavs):
        path = uav.path
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        plt.plot(xs, ys, color=colors[i], linewidth=2, marker='o', markersize=4,
                 label=f'UAV {uav.id} ({uav.role})')
        plt.plot(xs[0], ys[0], 's', color=colors[i], markersize=10)  # Start
        plt.plot(xs[-1], ys[-1], 's', color=colors[i], markersize=10)  # Goal

    plt.title("Multi-UAV Collaborative Path Planning (Comm-Aware)")
    plt.xlabel("X Grid")
    plt.ylabel("Y Grid")
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.xlim(0, MAP_WIDTH)
    plt.ylim(0, MAP_HEIGHT)

    plt.savefig("results/swarm_collaboration.png", dpi=300, bbox_inches='tight')
    print("✅ 多机协同图已保存: results/swarm_collaboration.png")
    plt.close()

    # 6. 保存 CSV 指标
    metrics = [
        {"UAV_ID": u.id, "Role": u.role, "Path_Length": len(u.path)}
        for u in uavs
    ]
    with open("results/swarm_metrics.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["UAV_ID", "Role", "Path_Length"])
        writer.writeheader()
        writer.writerows(metrics)

    print(f"\n📊 协同仿真结果:")
    print(f"  最小机间距离: {min_inter_dist:.2f} 格 (≥1.41 表示无碰撞)")
    print(f"  最大任务耗时: {max_path_len} 步")
    print(f"  角色分配: {[u.role for u in uavs]}")
    print("✅ 多无人机协同仿真完成！")


if __name__ == "__main__":
    main()