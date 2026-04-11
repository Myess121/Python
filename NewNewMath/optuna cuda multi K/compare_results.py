# compare_results.py
# -*- coding: utf-8 -*-
"""
对比不同K值的仿真结果，生成详细对比报告
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

dt = 20.0 / 180.0


def load_all_results():
    """加载所有K值的结果"""
    results = {}

    for k in range(2, 10):
        result_dir = Path(f"results_K{k}")
        if result_dir.exists():
            json_files = list(result_dir.glob("results_K*.json"))
            if json_files:
                latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results[k] = data

    return results


def plot_comprehensive_comparison(results):
    """绘制综合对比图"""
    fig = plt.figure(figsize=(16, 12))

    K_list = sorted(results.keys())
    n_k = len(K_list)

    # 提取数据
    best_times = []
    rate_10h = []
    rate_20h = []
    rate_40h = []
    all_best_trials = []

    for k in K_list:
        data = results[k]
        best_time = data['best_value']
        best_times.append(best_time)

        # 找到最佳试验
        best_trial = min(data['trials'], key=lambda x: x['results']['avg_time_hours'])
        all_best_trials.append(best_trial)
        rate_10h.append(best_trial['results']['detection_rate_10h'])
        rate_20h.append(best_trial['results']['detection_rate_20h'])
        rate_40h.append(best_trial['results']['detection_rate_40h'])

    # 1. 发现时间 vs K (左上)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(K_list, best_times, 'bo-', linewidth=2, markersize=10, markerfacecolor='red')
    ax1.fill_between(K_list, best_times, alpha=0.2)
    ax1.set_xlabel('无人机数量 K', fontsize=12)
    ax1.set_ylabel('平均发现时间 (小时)', fontsize=12)
    ax1.set_title('发现时间随无人机数量变化', fontsize=14)
    ax1.grid(True, alpha=0.3)

    for x, y in zip(K_list, best_times):
        ax1.annotate(f'{y:.2f}h', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

    # 2. 发现率对比 (右上)
    ax2 = fig.add_subplot(2, 2, 2)
    x = np.arange(len(K_list))
    width = 0.25

    bars1 = ax2.bar(x - width, rate_10h, width, label='10小时发现率', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x, rate_20h, width, label='20小时发现率', color='coral', alpha=0.8)
    bars3 = ax2.bar(x + width, rate_40h, width, label='40小时发现率', color='green', alpha=0.8)

    ax2.set_xlabel('无人机数量 K', fontsize=12)
    ax2.set_ylabel('发现率 (%)', fontsize=12)
    ax2.set_title('不同K值的发现率对比', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(K_list)
    ax2.legend()
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{height:.0f}%', ha='center', va='bottom', fontsize=8)

    # 3. 边际效益 (左下)
    ax3 = fig.add_subplot(2, 2, 3)
    if len(best_times) > 1:
        marginal_gains = []
        for i in range(1, len(best_times)):
            gain = best_times[i - 1] - best_times[i]
            marginal_gains.append(gain)

        colors = ['green' if g > 0 else 'red' for g in marginal_gains]
        ax3.bar(K_list[1:], marginal_gains, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('无人机数量 K', fontsize=12)
        ax3.set_ylabel('发现时间减少量 (小时)', fontsize=12)
        ax3.set_title('边际效益分析 (增加无人机的收益)', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        for i, (k, gain) in enumerate(zip(K_list[1:], marginal_gains)):
            ax3.annotate(f'{gain:.2f}h', (k, gain),
                         textcoords="offset points", xytext=(0, 5 if gain > 0 else -15),
                         ha='center', fontsize=9)

    # 4. 参数变化 (右下)
    ax4 = fig.add_subplot(2, 2, 4)

    param_names = ['H', 'pop_size', 'max_iter', 'penalty_coef']
    param_data = {p: [] for p in param_names}

    for k in K_list:
        data = results[k]
        params = data['best_params']
        for p in param_names:
            param_data[p].append(params.get(p, 0))

    x = np.arange(len(K_list))
    for p in param_names:
        ax4.plot(K_list, param_data[p], 'o-', label=p, linewidth=2, markersize=8)

    ax4.set_xlabel('无人机数量 K', fontsize=12)
    ax4.set_ylabel('参数值', fontsize=12)
    ax4.set_title('最优参数随K值变化趋势', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('无人机协同搜索 - 不同K值性能对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def generate_detailed_report(results):
    """生成详细报告"""
    print("\n" + "=" * 100)
    print(" " * 30 + "无人机协同搜索 - 详细分析报告")
    print("=" * 100)

    K_list = sorted(results.keys())

    # 创建数据表格
    data = []
    for k in K_list:
        data_item = results[k]
        best_trial = min(data_item['trials'], key=lambda x: x['results']['avg_time_hours'])

        # 计算各分位数
        all_times = best_trial['results']['all_steps']
        times_hours = np.array(all_times) * dt

        row = {
            'K': k,
            '最佳平均时间(h)': round(data_item['best_value'], 2),
            '中位数(h)': round(np.median(times_hours), 2),
            '标准差': round(np.std(times_hours), 2),
            '最小值(h)': round(np.min(times_hours), 2),
            '最大值(h)': round(np.max(times_hours), 2),
            '10h发现率(%)': round(best_trial['results']['detection_rate_10h'], 1),
            '20h发现率(%)': round(best_trial['results']['detection_rate_20h'], 1),
            '40h发现率(%)': round(best_trial['results']['detection_rate_40h'], 1),
        }
        data.append(row)

    df = pd.DataFrame(data)

    print("\n1. 性能汇总表")
    print("-" * 80)
    print(df.to_string(index=False))

    print("\n2. 最优参数配置")
    print("-" * 80)
    for k in K_list:
        print(f"\nK = {k}:")
        params = results[k]['best_params']
        for param, value in params.items():
            print(f"    {param}: {value}")

    print("\n3. 效率分析")
    print("-" * 80)

    # 计算增量收益
    base_time = results[K_list[0]]['best_value']
    print(f"\n以 K={K_list[0]} 为基准:")
    for k in K_list[1:]:
        current_time = results[k]['best_value']
        time_saved = base_time - current_time
        time_saved_per_uav = time_saved / (k - K_list[0])
        print(f"  K={k}: 节省 {time_saved:.2f}h, 每增加1架无人机节省 {time_saved_per_uav:.2f}h")

    # 计算边际收益递减
    print("\n边际收益递减分析:")
    for i in range(1, len(K_list)):
        gain = results[K_list[i - 1]]['best_value'] - results[K_list[i]]['best_value']
        print(f"  {K_list[i - 1]}→{K_list[i]}: 增益 {gain:.2f}h")

    print("\n" + "=" * 100)


def main():
    """主函数"""
    print("加载所有K值的结果...")
    results = load_all_results()

    if not results:
        print("未找到任何结果文件！")
        print("请先运行 main_tuning.py 进行调参。")
        return

    print(f"找到 {len(results)} 个K值的结果: {sorted(results.keys())}")

    # 生成对比图
    plot_comprehensive_comparison(results)

    # 生成详细报告
    generate_detailed_report(results)

    # 保存汇总数据
    save_summary_data(results)


def save_summary_data(results):
    """保存汇总数据到CSV"""
    K_list = sorted(results.keys())
    data = []

    for k in K_list:
        data_item = results[k]
        best_trial = min(data_item['trials'], key=lambda x: x['results']['avg_time_hours'])

        row = {
            'K': k,
            'avg_time_hours': round(data_item['best_value'], 2),
            'median_hours': round(np.median(np.array(best_trial['results']['all_steps']) * dt), 2),
            'std_hours': round(np.std(np.array(best_trial['results']['all_steps']) * dt), 2),
            'detection_rate_10h': round(best_trial['results']['detection_rate_10h'], 1),
            'detection_rate_20h': round(best_trial['results']['detection_rate_20h'], 1),
            'detection_rate_40h': round(best_trial['results']['detection_rate_40h'], 1),
        }

        # 添加参数
        for param, value in data_item['best_params'].items():
            if isinstance(value, float):
                row[param] = round(value, 4)
            else:
                row[param] = value

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv('summary_all_K.csv', index=False, encoding='utf-8-sig')
    print("\n汇总数据已保存到: summary_all_K.csv")


if __name__ == "__main__":
    main()