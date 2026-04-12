# run_swarm_experiment.py
"""
多无人机协同实验脚本
验证：任务分配 + 协同规划 + 防碰撞
"""
from environment import GridMap, GroundStation, Jammer
from swarm.uav import UAV
from swarm.swarm_planner import SwarmPlanner
from config import *
import utils
import matplotlib.pyplot as plt


def main():
    print("🚀 开始多无人机协同实验...")

    # 1. 初始化环境
    grid_map = GridMap()
    gs = GroundStation()
    jammers = [Jammer(pos=(25.0, 15.0))]

    # 2. 创建 3 架无人机
    uavs = [
        UAV(uav_id=0, start_pos=(5, 5), goal_pos=(45, 25), role="scout"),
        UAV(uav_id=1, start_pos=(5, 15), goal_pos=(45, 15), role="relay"),
        UAV(uav_id=2, start_pos=(5, 25), goal_pos=(45, 5), role="scout")
    ]

    # 3. 协同规划
    planner = SwarmPlanner(uavs, grid_map, gs, jammers)
    all_paths = planner.plan_collaboratively()

    # 4. 可视化
    plt.figure(figsize=(10, 6))

    # 画障碍物
    if grid_map.obstacles:
        obs_x = [p[0] for p in grid_map.obstacles]
        obs_y = [p[1] for p in grid_map.obstacles]
        plt.plot(obs_x, obs_y, 'k^', markersize=8, label='Obstacles')

    # 画干扰源
    if jammers:
        plt.plot(jammers[0].pos[0], jammers[0].pos[1], 'r*', markersize=20, label='Jammer')

    # 画每架无人机的路径
    colors = ['blue', 'green', 'orange']
    for i, uav in enumerate(uavs):
        path = all_paths[uav.id]
        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            plt.plot(xs, ys, color=colors[i % len(colors)], linewidth=2,
                     marker='o', markersize=4, label=f'UAV {uav.id} ({uav.role})')
            plt.plot(xs[0], ys[0], 's', color=colors[i], markersize=10)  # Start
            plt.plot(xs[-1], ys[-1], 's', color=colors[i], markersize=10)  # Goal

    plt.title("Multi-UAV Collaborative Path Planning")
    plt.xlabel("X Grid")
    plt.ylabel("Y Grid")
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, MAP_WIDTH)
    plt.ylim(0, MAP_HEIGHT)
    plt.savefig("results/swarm_planning.png", dpi=300, bbox_inches='tight')
    print("✅ 多无人机协同图已保存：results/swarm_planning.png")
    plt.show()

    # 5. 打印统计
    print("\n📊 多无人机实验结果:")
    for uav in uavs:
        print(f"  UAV {uav.id} ({uav.role}): 路径长度 = {len(all_paths[uav.id])}")


if __name__ == "__main__":
    main()