# run_experiment.py
"""
完整对比实验脚本：A* vs D* Lite vs Comm-Aware D* Lite
自动采集指标、生成对比图、SNR时序图、CSV表格
"""
import os
import time
import csv
import matplotlib.pyplot as plt

from config import *
from environment import GridMap, GroundStation, Jammer, calculate_snr
from planners.astar import AStar
from planners.dstar_lite import DStarLite
from planners.dstar_comm import DStarComm
import utils


def run_and_record(planner, grid_map, gs, jammers, name):
    """运行单次规划并记录指标"""
    start_time = time.time()
    path = planner.plan(grid_map, gs, jammers)
    run_time = time.time() - start_time

    snr_sequence = []
    interrupt_steps = 0
    for pos in path:
        snr = calculate_snr(pos, gs, jammers)
        snr_sequence.append(snr)
        if snr < SNR_THRESHOLD:
            interrupt_steps += 1

    metrics = {
        "Algorithm": name,
        "Path_Length": len(path),
        "Min_SNR_dB": min(snr_sequence) if snr_sequence else 0,
        "Interrupt_Steps": interrupt_steps,
        "Run_Time_s": round(run_time, 4),
        "Path_Coords": str(path),
        "SNR_Sequence": str(snr_sequence)
    }
    return path, snr_sequence, metrics


def plot_snr_timeline(results_dict, save_name="results/snr_timeline.png"):
    """绘制 SNR 时序对比图"""
    plt.figure(figsize=(8, 4))
    colors = {'A*': 'blue', 'Standard D* Lite': 'green', 'Comm-Aware D* Lite': 'orange'}
    for name, data in results_dict.items():
        plt.plot(data['SNR'], label=name, color=colors.get(name, 'gray'), linewidth=2)
    plt.axhline(y=SNR_THRESHOLD, color='red', linestyle='--', label=f'LQM Threshold ({SNR_THRESHOLD})')
    plt.title("SNR Variation Along Planned Paths")
    plt.xlabel("Path Step")
    plt.ylabel("LQM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✅ SNR时序图已保存: {save_name}")
    plt.close()


def save_to_csv(all_metrics, filename="results/metrics.csv"):
    """保存对比表格到CSV"""
    if not os.path.exists("results"):
        os.makedirs("results")
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["Algorithm", "Path_Length", "Min_SNR_dB", "Interrupt_Steps", "Run_Time_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"✅ 数据表格已保存: {filename}")



def main():
    print("🚀 开始运行完整对比实验 (含 A* 基线)...")
    os.makedirs("results", exist_ok=True)

    grid_map = GridMap()
    gs = GroundStation()
    jammers = [Jammer()]

    # 1. A* (静态基线)
    print("⏳ 正在计算: A* ...")
    astar = AStar(START_POS, GOAL_POS)
    path_a, snr_a, met_a = run_and_record(astar, grid_map, gs, jammers, "A*")

    # 2. D* Lite (动态基线)
    print("⏳ 正在计算: Standard D* Lite...")
    dstar = DStarLite(START_POS, GOAL_POS)
    path_d, snr_d, met_d = run_and_record(dstar, grid_map, gs, jammers, "Standard D* Lite")

    # 3. Comm-Aware D* Lite (本文方法)
    print("⏳ 正在计算: Comm-Aware D* Lite...")
    dstar_comm = DStarComm(START_POS, GOAL_POS)
    path_c, snr_c, met_c = run_and_record(dstar_comm, grid_map, gs, jammers, "Comm-Aware D* Lite")

    results = {
        "A*": {"Path": path_a, "SNR": snr_a},
        "Standard D* Lite": {"Path": path_d, "SNR": snr_d},
        "Comm-Aware D* Lite": {"Path": path_c, "SNR": snr_c}
    }
    all_metrics = [met_a, met_d, met_c]

    # 1. 画轨迹对比图
    utils.plot_comparison(grid_map, jammers, {k: v["Path"] for k, v in results.items()})

    # 2. 画 SNR 时序图
    plot_snr_timeline(results)

    # 3. 保存数据表
    save_to_csv(all_metrics)

    # 4. 终端打印摘要
    print("\n📊 实验结果摘要:")
    print(f"{'算法':<20} | {'路径长度':<6} | {'最低SNR(dB)':<8} | {'中断步数':<6} | {'耗时(s)':<6}")
    print("-" * 65)
    for m in all_metrics:
        print(
            f"{m['Algorithm']:<20} | {m['Path_Length']:<6} | {m['Min_SNR_dB']:<8.2f} | {m['Interrupt_Steps']:<6} | {m['Run_Time_s']:<6}")


if __name__ == "__main__":
    main()