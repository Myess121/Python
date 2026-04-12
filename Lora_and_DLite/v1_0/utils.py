# utils.py
"""
可视化工具集
包含：静态对比图、动态演化图、SNR 时序图
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import *


def plot_comparison(grid_map, jammers, paths_dict, save_name="results/comparison.png"):
    """
    绘制静态轨迹对比图（用于 run_experiment.py）
    """
    import os
    if not os.path.exists("results"):
        os.makedirs("results")

    plt.figure(figsize=(10, 6))

    # 1. 画障碍物
    if grid_map.obstacles:
        obs_x = [p[0] for p in grid_map.obstacles]
        obs_y = [p[1] for p in grid_map.obstacles]
        plt.plot(obs_x, obs_y, 'k^', markersize=8, label='Obstacles')

    # 2. 画干扰源
    if jammers:
        for i, jammer in enumerate(jammers):
            plt.plot(jammer.pos[0], jammer.pos[1], 'r*', markersize=20, label='Jammer' if i == 0 else "")
            # 画干扰范围虚线圈
            circle = plt.Circle((jammer.pos[0], jammer.pos[1]), 10, color='r', fill=False, linestyle='--', alpha=0.3)
            plt.gca().add_patch(circle)

    # 3. 画路径
    colors = ['blue', 'green', 'orange', 'purple']
    for i, (name, path) in enumerate(paths_dict.items()):
        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            plt.plot(xs, ys, color=colors[i % len(colors)], linewidth=2, marker='o', markersize=4, label=name)

    # 4. 标注起点终点
    plt.plot(START_POS[0], START_POS[1], 'gs', markersize=15, label='Start')
    plt.plot(GOAL_POS[0], GOAL_POS[1], 'rs', markersize=15, label='Goal')

    plt.title("Path Planning Comparison: Standard vs. Comm-Aware D* Lite")
    plt.xlabel("X Grid")
    plt.ylabel("Y Grid")
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, MAP_WIDTH)
    plt.ylim(0, MAP_HEIGHT)

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已保存至: {save_name}")
    plt.close()



def plot_dynamic_evolution(frames_data, grid_map, save_name="results/dynamic_evolution.png"):
    """
    绘制动态重规划演化图（用于 run_dynamic_scenario.py）
    """
    n_frames = len(frames_data)
    cols = 3
    rows = (n_frames + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))

    for i, data in enumerate(frames_data):
        ax = plt.subplot(rows, cols, i + 1)

        # 1. 画背景 (障碍物)
        if grid_map.obstacles:
            obs_x = [p[0] for p in grid_map.obstacles]
            obs_y = [p[1] for p in grid_map.obstacles]
            ax.plot(obs_x, obs_y, 'k^', markersize=5)

        # 2. 画当前干扰源
        jammer_pos = data['jammer_pos']
        ax.plot(jammer_pos[0], jammer_pos[1], 'r*', markersize=15, label='Jammer')

        # 3. 画当前路径
        path = data['path']
        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, 'g-o', linewidth=2, markersize=4, label='Planned Path')
            ax.plot(xs[0], ys[0], 'gs', markersize=10)  # Start
            ax.plot(xs[-1], ys[-1], 'rs', markersize=10)  # Goal

        ax.set_title(f"Time Step: {data['time']}")
        ax.set_xlim(0, MAP_WIDTH)
        ax.set_ylim(0, MAP_HEIGHT)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✅ 动态演化图已保存: {save_name}")
    plt.close()