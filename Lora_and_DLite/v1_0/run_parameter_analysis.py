# run_parameter_analysis.py
"""
参数敏感性分析
测试不同COMM_WEIGHT对路径长度和SNR的影响
"""
import matplotlib.pyplot as plt
from config import *


def main():
    # 测试不同权重
    comm_weights = [5, 10, 20, 50, 100]
    path_lengths = []
    min_snrs = []

    for weight in comm_weights:
        # 修改config.COMM_WEIGHT
        # 运行实验
        # 记录指标
        pass

    # 画权衡曲线
    plt.figure()
    plt.plot(comm_weights, path_lengths, 'o-', label='Path Length')
    plt.plot(comm_weights, min_snrs, 's-', label='Min SNR')
    plt.xlabel('COMM_WEIGHT')
    plt.legend()
    plt.savefig("results/parameter_sensitivity.png")