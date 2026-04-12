# run_monte_carlo.py
"""
蒙特卡洛实验：随机场景统计
生成20组随机干扰源位置/速度，统计性能指标
"""
import numpy as np
import random
from environment import GridMap, GroundStation, Jammer
from planners.dstar_lite import DStarLite
from planners.dstar_comm import DStarComm
from config import *


def generate_random_scenario():
    """生成随机场景"""
    # 随机干扰源位置
    jammer_pos = (random.uniform(15, 35), random.uniform(10, 20))
    jammer_vel = (random.uniform(0.5, 1.5), random.uniform(-0.5, 0.5))

    return Jammer(pos=jammer_pos, velocity=jammer_vel)


def main():
    n_runs = 20
    results = {
        "Standard_DLite": {"length": [], "min_snr": [], "interrupt": []},
        "Comm_Aware_DLite": {"length": [], "min_snr": [], "interrupt": []}
    }

    for i in range(n_runs):
        print(f"Run {i + 1}/{n_runs}")
        grid_map = GridMap()
        gs = GroundStation()
        jammer = generate_random_scenario()

        # 运行两种算法
        # ... 记录指标 ...

    # 输出统计结果（均值±标准差）
    print("\n📊 蒙特卡洛统计结果:")
    for algo, metrics in results.items():
        print(f"{algo}:")
        print(f"  路径长度: {np.mean(metrics['length']):.1f} ± {np.std(metrics['length']):.1f}")
        print(f"  最低SNR: {np.mean(metrics['min_snr']):.1f} ± {np.std(metrics['min_snr']):.1f} dB")


if __name__ == "__main__":
    main()