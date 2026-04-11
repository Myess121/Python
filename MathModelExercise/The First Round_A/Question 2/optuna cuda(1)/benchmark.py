# -*- coding: utf-8 -*-
"""
CPU vs GPU 性能对比测试
"""

import time
import numpy as np
from uav_simulator import UAVSimulatorCUDA
from uav_simulator_cpu import UAVSimulatorCPU  # 原CPU版本


def benchmark():
    """性能对比测试"""

    # 测试参数
    K = 7
    params = {
        'H': 6,
        'pop_size': 12,
        'max_iter': 20,
        'w_start': 0.9,
        'w_end': 0.4,
        'c1': 1.5,
        'c2': 1.5,
        'penalty_coef': 1000.0
    }

    n_simulations = 50
    uav_start = [(0, 0)] * K
    init_positions = [(np.random.randint(0, 16), np.random.randint(0, 23))
                      for _ in range(n_simulations)]

    print("=" * 60)
    print("CPU vs GPU 性能对比")
    print("=" * 60)
    print(f"仿真次数: {n_simulations}")
    print(f"无人机数量: {K}")
    print(f"参数: H={params['H']}, pop_size={params['pop_size']}, max_iter={params['max_iter']}")
    print("=" * 60)

    # CPU测试
    print("\n1. CPU版本测试...")
    cpu_sim = UAVSimulatorCPU()
    start_time = time.time()
    cpu_results = []
    for init_pos in init_positions:
        result = cpu_sim.run_simulation(K, init_pos, uav_start, params)
        cpu_results.append(result)
    cpu_time = time.time() - start_time
    cpu_avg_time = np.mean(cpu_results) * (20 / 180)  # dt转换

    print(f"   CPU耗时: {cpu_time:.2f} 秒")
    print(f"   平均发现时间: {cpu_avg_time:.2f} 小时")

    # GPU测试
    print("\n2. GPU版本测试...")
    gpu_sim = UAVSimulatorCUDA()
    start_time = time.time()
    gpu_results = gpu_sim.batch_simulate(K, init_positions, uav_start, params, n_processes=1)
    gpu_time = time.time() - start_time
    gpu_avg_time = np.mean(gpu_results) * (20 / 180)

    print(f"   GPU耗时: {gpu_time:.2f} 秒")
    print(f"   平均发现时间: {gpu_avg_time:.2f} 小时")

    # 加速比
    speedup = cpu_time / gpu_time
    print("\n" + "=" * 60)
    print(f"加速比: {speedup:.2f}x")
    print(f"性能提升: {(speedup - 1) * 100:.1f}%")
    print("=" * 60)

    return speedup


if __name__ == "__main__":
    benchmark()