# analyze_all_results.py
# -*- coding: utf-8 -*-
"""
分析所有K值的结果，生成对比图表
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 时间步长
dt = 20.0 / 180.0  # 小时/步


class MultiKAnalyzer:
    """多K值结果分析器"""

    def __init__(self, summary_file=None):
        """初始化分析器"""
        if summary_file and Path(summary_file).exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                self.summary = json.load(f)
        else:
            # 查找所有结果文件
            self.summary = self.load_all_results()

    def load_all_results(self):
        """加载所有K值的结果"""
        results = []

        # 查找所有 results_K* 目录
        for result_dir in Path(".").glob("results_K*"):
            if result_dir.is_dir():
                # 查找最新的结果文件
                json_files = list(result_dir.glob("results_K*.json"))
                if json_files:
                    # 选择最新的文件
                    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        results.append({
                            'K': data['K'],
                            'best_params': data['best_params'],
                            'best_value': data['best_value'],
                            'trials': data['trials']
                        })

        # 按K值排序
        results.sort(key=lambda x: x['K'])

        return {'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"), 'results': results}

    def plot_performance_comparison(self, save_path=None):
        """绘制性能对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        K_list = [r['K'] for r in self.summary['results']]
        best_times = [r['best_value'] for r in self.summary['results']]

        # 提取各试验的发现率
        rate_10h = []
        rate_20h = []
        rate_40h = []

        for r in self.summary['results']:
            # 从trials中找最佳试验的结果
            best_trial = min(r['trials'], key=lambda x: x['results']['avg_time_hours'])
            rate_10h.append(best_trial['results']['detection_rate_10h'])
            rate_20h.append(best_trial['results']['detection_rate_20h'])
            rate_40h.append(best_trial['results']['detection_rate_40h'])

        # 1. 平均发现时间 vs K
        ax1 = axes[0, 0]
        ax1.plot(K_list, best_times, 'bo-', linewidth=2, markersize=8, markerfacecolor='red')
        ax1.fill_between(K_list, best_times, alpha=0.3)
        ax1.set_xlabel('无人机数量 K')
        ax1.set_ylabel('最佳平均发现时间 (小时)')
        ax1.set_title('无人机数量 vs 发现时间')
        ax1.grid(True, alpha=0.3)

        # 添加数值标签
        for x, y in zip(K_list, best_times):
            ax1.annotate(f'{y:.1f}h', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        # 2. 发现率对比
        ax2 = axes[0, 1]
        x = np.arange(len(K_list))
        width = 0.25

        ax2.bar(x - width, rate_10h, width, label='10小时发现率', color='steelblue', alpha=0.8)
        ax2.bar(x, rate_20h, width, label='20小时发现率', color='coral', alpha=0.8)
        ax2.bar(x + width, rate_40h, width, label='40小时发现率', color='green', alpha=0.8)

        ax2.set_xlabel('无人机数量 K')
        ax2.set_ylabel('发现率 (%)')
        ax2.set_title('不同K值的发现率对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(K_list)
        ax2.legend()
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. 边际效益分析 (发现时间减少量)
        ax3 = axes[1, 0]
        if len(best_times) > 1:
            marginal_gains = []
            for i in range(1, len(best_times)):
                gain = best_times[i - 1] - best_times[i]
                marginal_gains.append(gain)

            ax3.bar(K_list[1:], marginal_gains, color='purple', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('无人机数量增加 (K-1 → K)')
            ax3.set_ylabel('发现时间减少量 (小时)')
            ax3.set_title('边际效益分析')
            ax3.grid(True, alpha=0.3, axis='y')

            for i, gain in enumerate(marginal_gains):
                ax3.annotate(f'{gain:.1f}h', (K_list[i + 1], gain),
                             textcoords="offset points", xytext=(0, 5), ha='center')

        # 4. 参数变化趋势
        ax4 = axes[1, 1]
        param_names = ['H', 'pop_size', 'max_iter', 'penalty_coef']
        param_data = {p: [] for p in param_names}

        for r in self.summary['results']:
            params = r['best_params']
            for p in param_names:
                param_data[p].append(params.get(p, 0))

        x = np.arange(len(K_list))
        for p in param_names:
            ax4.plot(K_list, param_data[p], 'o-', label=p, linewidth=2, markersize=6)

        ax4.set_xlabel('无人机数量 K')
        ax4.set_ylabel('参数值')
        ax4.set_title('最优参数随K值变化趋势')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"性能对比图已保存: {save_path}")
        plt.show()

    def plot_k_distributions(self, save_path=None):
        """绘制各K值的发现时间分布"""
        n_k = len(self.summary['results'])
        n_cols = min(4, n_k)
        n_rows = (n_k + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, r in enumerate(self.summary['results']):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # 提取所有试验的最佳结果
            best_times = []
            for trial in r['trials']:
                if 'all_times' in trial['results']:
                    best_times.extend(trial['results']['all_times'])
                else:
                    # 如果没有all_times，使用avg_time
                    best_times.append(trial['results']['avg_time_hours'])

            ax.hist(best_times, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(r['best_value'], color='red', linestyle='--',
                       label=f'最佳: {r["best_value"]:.1f}h')
            ax.set_xlabel('发现时间 (小时)')
            ax.set_ylabel('频次')
            ax.set_title(f'K = {r["K"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(len(self.summary['results']), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"分布图已保存: {save_path}")
        plt.show()

    def plot_parameter_heatmap(self, save_path=None):
        """绘制参数与K值的热力图"""
        K_list = [r['K'] for r in self.summary['results']]

        # 提取参数
        param_names = ['H', 'pop_size', 'max_iter', 'w_start', 'w_end', 'c1', 'c2', 'penalty_coef']
        param_matrix = []

        for r in self.summary['results']:
            params = r['best_params']
            row = [params.get(p, 0) for p in param_names]
            param_matrix.append(row)

        param_matrix = np.array(param_matrix)

        # 归一化处理
        param_matrix_norm = (param_matrix - param_matrix.min(axis=0)) / (
                    param_matrix.max(axis=0) - param_matrix.min(axis=0) + 1e-8)

        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(param_matrix_norm.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # 设置标签
        ax.set_xticks(np.arange(len(K_list)))
        ax.set_yticks(np.arange(len(param_names)))
        ax.set_xticklabels([f'K={k}' for k in K_list])
        ax.set_yticklabels(param_names)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 添加数值
        for i in range(len(param_names)):
            for j in range(len(K_list)):
                text = ax.text(j, i, f'{param_matrix[j, i]:.1f}',
                               ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=ax, label='归一化参数值')
        ax.set_title('不同K值的最优参数热力图')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"参数热力图已保存: {save_path}")
        plt.show()

    def generate_report(self, output_dir="analysis_multi_k"):
        """生成完整分析报告"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("\n生成分析图表...")

        self.plot_performance_comparison(output_path / "performance_comparison.png")
        self.plot_k_distributions(output_path / "k_distributions.png")
        self.plot_parameter_heatmap(output_path / "parameter_heatmap.png")

        # 保存汇总表格
        self.save_summary_table(output_path / "summary_table.csv")

        # 打印报告
        self.print_report()

        print(f"\n分析报告已保存到: {output_path}")

    def save_summary_table(self, save_path):
        """保存汇总表格"""
        data = []
        for r in self.summary['results']:
            best_trial = min(r['trials'], key=lambda x: x['results']['avg_time_hours'])
            row = {
                'K': r['K'],
                '最佳平均时间(h)': round(r['best_value'], 2),
                '10h发现率(%)': round(best_trial['results']['detection_rate_10h'], 1),
                '20h发现率(%)': round(best_trial['results']['detection_rate_20h'], 1),
                '40h发现率(%)': round(best_trial['results']['detection_rate_40h'], 1),
                'H': r['best_params'].get('H', '-'),
                'pop_size': r['best_params'].get('pop_size', '-'),
                'max_iter': r['best_params'].get('max_iter', '-'),
                'w_start': round(r['best_params'].get('w_start', 0), 4),
                'w_end': round(r['best_params'].get('w_end', 0), 4),
                'c1': round(r['best_params'].get('c1', 0), 4),
                'c2': round(r['best_params'].get('c2', 0), 4),
                'penalty_coef': round(r['best_params'].get('penalty_coef', 0), 1)
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"汇总表格已保存: {save_path}")

    def print_report(self):
        """打印分析报告"""
        print("\n" + "=" * 80)
        print("多无人机数量调参分析报告")
        print("=" * 80)

        print("\n各K值最佳结果汇总:")
        print("-" * 80)
        print(f"{'K':<4} {'最佳时间(h)':<12} {'10h率(%)':<10} {'20h率(%)':<10} {'40h率(%)':<10}")
        print("-" * 80)

        for r in self.summary['results']:
            best_trial = min(r['trials'], key=lambda x: x['results']['avg_time_hours'])
            print(f"{r['K']:<4} {r['best_value']:<12.2f} "
                  f"{best_trial['results']['detection_rate_10h']:<10.1f} "
                  f"{best_trial['results']['detection_rate_20h']:<10.1f} "
                  f"{best_trial['results']['detection_rate_40h']:<10.1f}")

        print("-" * 80)

        # 计算效率指标
        print("\n效率分析 (每架无人机的贡献):")
        print("-" * 60)

        results_list = self.summary['results']
        if len(results_list) >= 2:
            base_time = results_list[0]['best_value']
            for r in results_list:
                k = r['K']
                time_saved = base_time - r['best_value'] if k > results_list[0]['K'] else 0
                efficiency = (base_time - r['best_value']) / (k - results_list[0]['K']) if k > results_list[0][
                    'K'] else 0
                print(f"K={k}: 相比K=2节省 {time_saved:.2f}h, 每增加1架无人机平均节省 {efficiency:.2f}h")

        print("=" * 80)


def main():
    """主函数"""
    print("\n多无人机数量结果分析器")
    print("=" * 60)

    # 查找汇总文件
    summary_file = Path("summary_all_K.json")

    if not summary_file.exists():
        print("未找到 summary_all_K.json，正在从各结果目录加载...")

    analyzer = MultiKAnalyzer(summary_file if summary_file.exists() else None)
    analyzer.generate_report()


if __name__ == "__main__":
    from datetime import datetime

    main()