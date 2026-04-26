import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统黑体
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('Capstone Data.csv')

titles = [
    "单缝衍射 a=0.02mm (运行#1)",
    "单缝衍射 a=0.04mm (运行#2)",
    "双缝干涉 a=0.04mm, d=0.25mm (运行#3)",
    "双缝干涉 a=0.04mm, d=0.50mm (运行#4)"
]

for i in range(1, 5):
    x_col = f'位置 (米 m) 运行#{i}'
    y_col = f'相对强度 运行#{i}'
    data = df[[x_col, y_col]].dropna()
    x = data[x_col].values
    y = data[y_col].values

    # 归一化
    y_norm = y / np.max(y)

    plt.figure(figsize=(12, 6))
    plt.plot(x, y_norm, color='black', linewidth=1.0)

    if i >= 3:
        # 寻找所有的峰
        peaks, _ = find_peaks(y_norm, height=0.01, prominence=0.01, distance=10)
        peak_x = x[peaks]
        peak_y = y_norm[peaks]

        # 找到最高的中心峰作为0级
        center_idx = np.argmax(peak_y)
        center_pos = peak_x[center_idx]

        # 精确物理间距
        delta_x = 0.00221 if i == 3 else 0.001114

        labeled_orders = set()

        # 设定最大标注级次 (第3张图到16，第4张图到22)
        max_order = 16 if i == 3 else 22

        # 遍历所有找到的峰
        for px, py in zip(peak_x, peak_y):
            # 计算级次
            order = int(np.round((px - center_pos) / delta_x))
            abs_order = abs(order)

            # 过滤掉不需要的级次
            if abs_order <= max_order and py > 0.02 and order not in labeled_orders:
                plt.text(px, py + 0.02, str(abs_order),
                         ha='center', va='bottom', fontsize=10, color='blue')
                labeled_orders.add(order)

        # 自动寻找并标注“缺级”
        for order in range(-max_order, max_order + 1):
            expected_x = center_pos + order * delta_x
            if min(x) + 0.005 < expected_x < max(x) - 0.005:
                # 如果该级次没出现，说明是缺级
                if order not in labeled_orders and abs(order) > 0:
                    idx_nearest = np.argmin(np.abs(x - expected_x))
                    if y_norm[idx_nearest] < 0.15:
                        abs_order = abs(order)
                        # 【修改点】高度从 0.02 提高到了 0.08，不压线！
                        plt.text(expected_x, 0.08, f"{abs_order}",
                                 ha='center', va='bottom', fontsize=10, color='blue')

        # 动态裁剪坐标轴，把画面左右的空白去掉
        plt.xlim(center_pos - max_order * delta_x - 0.006, center_pos + max_order * delta_x + 0.006)
    else:
        # 单缝图的坐标范围
        plt.xlim(min(x), max(x))

    # 完善图表信息
    plt.title(titles[i - 1], fontsize=15)
    plt.xlabel('位置 (m)', fontsize=12)
    plt.ylabel('相对强度', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_name = f'高分版_图_{i}.png'
    plt.savefig(save_name, dpi=300)
    print(f"已生成并保存: {save_name}")