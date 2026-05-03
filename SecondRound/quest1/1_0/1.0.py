# -*- coding: utf-8 -*-
"""
第一问：国一封神版 (离散时间窗提取 + 全维时间融合 + GCV验证 + 2D交互偏依赖图)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pygam import LinearGAM, s, te, f
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 辅助函数：格式化断点滞后 =====================
def format_lags(lags):
    """将离散的滞后列表转化为连续区间段，例如 [1,2,3,7,8] -> '1-3h, 7-8h'"""
    if not lags: return "无显著滞后"
    lags = sorted(lags)
    ranges = []
    start = lags[0]
    for i in range(1, len(lags)):
        if lags[i] != lags[i - 1] + 1:
            ranges.append(f"{start}-{lags[i - 1]}h" if start != lags[i - 1] else f"{start}h")
            start = lags[i]
    ranges.append(f"{start}-{lags[-1]}h" if start != lags[-1] else f"{start}h")
    return ", ".join(ranges)


# ===================== 1. 数据读取与全维时间提取 =====================
print("正在读取数据并提取全维度时间特征...")
df = pd.read_excel("4-2026年校赛第二轮A题数据.xlsx", sheet_name="2026年1-3月良乡空气质量数据", skiprows=9)

# 【核心修复】：加上 format='%Y-%m-%d%H:%M:%S'，让 pandas 能够认出无空格的时间字符串
df['datetime'] = pd.to_datetime(df['datetime'].astype(str).str.replace(" ", ""), format='%Y-%m-%d%H:%M:%S', errors='coerce')
df = df.dropna(subset=['datetime']).reset_index(drop=True)

# 提取全维时间特征
df['hour'] = df['datetime'].dt.hour
df['dayofyear'] = df['datetime'].dt.dayofyear
df['weekday'] = df['datetime'].dt.weekday  # 0-6 (周一至周日)

pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
weather = ['temperature', 'humidity']

for col in pollutants + weather:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].interpolate(method='linear', limit_direction='both').bfill().ffill()

# ===================== 2. Spearman 离散滞后断点分析 =====================
print("\n" + "=" * 60)
print("正在进行 Spearman 显著性滞后分析 (精准提取断点分布)...")
max_lag = 24
lag_info_dict = {}
lag_summary = []

for pol in pollutants:
    lag_info_dict[pol] = {}
    for wea in weather:
        best_lag, max_abs_corr, best_corr = 0, 0, 0
        significant_lags = []

        for lag in range(1, max_lag + 1):
            valid_df = df[[pol, wea]].copy()
            valid_df[f'{wea}_lag'] = valid_df[wea].shift(lag)
            valid_df = valid_df.dropna()

            corr, p_val = spearmanr(valid_df[pol], valid_df[f'{wea}_lag'])

            if p_val < 0.05:  # 仅记录显著(p<0.05)的滞后阶数
                significant_lags.append(lag)
                if abs(corr) > max_abs_corr:
                    max_abs_corr = abs(corr)
                    best_lag = lag
                    best_corr = corr

        window_str = format_lags(significant_lags)
        lag_info_dict[pol][wea] = best_lag

        lag_summary.append({
            '污染物': pol, '气象要素': '温度' if wea == 'temperature' else '湿度',
            '显著滞后分布段': window_str,
            '极值滞后点': f"滞后 {best_lag} 小时",
            '极值 Spearman': round(best_corr, 4)
        })

df_lag = pd.DataFrame(lag_summary)
df_lag.to_csv('Spearman_Discrete_Lags.csv', index=False, encoding='utf-8-sig')
print("断点滞后提取完成！已保存为 Spearman_Discrete_Lags.csv")

# ===================== 3. 全维时间 GAM 建模 & 2D协同可视化 =====================
print("\n" + "=" * 60)
print("正在构建 GAM，包含 GCV 泛化验证与 2D 协同等高线图...")

gam_results = []

for pol in pollutants:
    print(f"拟合 {pol} GAM 模型中...")
    best_temp_lag = lag_info_dict[pol]['temperature']
    best_hum_lag = lag_info_dict[pol]['humidity']

    # 动态构建数据集
    df_model = df[['hour', 'dayofyear', 'weekday', pol]].copy()
    df_model['temp_lag'] = df['temperature'].shift(best_temp_lag) if best_temp_lag > 0 else df['temperature']
    df_model['hum_lag'] = df['humidity'].shift(best_hum_lag) if best_hum_lag > 0 else df['humidity']
    df_model = df_model.dropna()

    # 特征 X: [0:hour, 1:dayofyear, 2:weekday, 3:temp, 4:hum]
    X = df_model[['hour', 'dayofyear', 'weekday', 'temp_lag', 'hum_lag']].values
    y = df_model[pol].values

    # 模型架构：日循环平滑 + 长期趋势平滑 + 星期因子 + 温度平滑 + 湿度平滑 + 温湿二维张量积
    # 使用 f() 处理分类变量 weekday
    gam = LinearGAM(s(0) + s(1) + f(2) + s(3) + s(4) + te(3, 4))
    gam.gridsearch(X, y, progress=False)

    # 提取统计量
    pseudo_r2 = gam.statistics_['pseudo_r2']['explained_deviance']
    gcv_score = gam.statistics_['GCV']  # 提取广义交叉验证得分

    gam_results.append({
        '污染物': pol,
        'Pseudo R²': round(pseudo_r2, 4),
        'GCV (广义交叉验证)': round(gcv_score, 4)
    })

    # ================= 绘制可视化图表 =================
    fig = plt.figure(figsize=(18, 5))

    # 图 1: 时间趋势 (Hour)
    ax1 = fig.add_subplot(1, 3, 1)
    XX_hour = gam.generate_X_grid(term=0)
    pdep_hour, conf_hour = gam.partial_dependence(term=0, X=XX_hour, width=.95)
    ax1.plot(XX_hour[:, 0], pdep_hour, color='purple', linewidth=2)
    ax1.fill_between(XX_hour[:, 0], conf_hour[:, 0], conf_hour[:, 1], color='purple', alpha=0.2)
    ax1.set_title("日际循环效应 (Hour)", fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xticks(range(0, 24, 4))

    # 图 2: 长期趋势 (Day of Year)
    ax2 = fig.add_subplot(1, 3, 2)
    XX_doy = gam.generate_X_grid(term=1)
    pdep_doy, conf_doy = gam.partial_dependence(term=1, X=XX_doy, width=.95)
    ax2.plot(XX_doy[:, 1], pdep_doy, color='green', linewidth=2)
    ax2.fill_between(XX_doy[:, 1], conf_doy[:, 0], conf_doy[:, 1], color='green', alpha=0.2)
    ax2.set_title("长期演变趋势 (1-3月)", fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 图 3: 温湿二维交互协同图 (核心亮点)
    ax3 = fig.add_subplot(1, 3, 3)
    term_idx_te = 5  # te(3, 4) 是第 6 个项 (索引 5)
    XX_te = gam.generate_X_grid(term=term_idx_te)
    Z = gam.partial_dependence(term=term_idx_te, X=XX_te)

    # pygam的张量积网格默认是 100x100
    x_grid = XX_te[:, 3].reshape(100, 100)
    y_grid = XX_te[:, 4].reshape(100, 100)
    z_grid = Z.reshape(100, 100)

    # 绘制等高线热力图
    contour = ax3.contourf(x_grid, y_grid, z_grid, levels=20, cmap='RdYlBu_r', alpha=0.8)
    fig.colorbar(contour, ax=ax3, label="协同偏依赖浓度")
    ax3.set_xlabel(f"温度 (滞后{best_temp_lag}h)")
    ax3.set_ylabel(f"湿度 (滞后{best_hum_lag}h)")
    ax3.set_title("温湿度 2D 交互协同偏依赖图", fontsize=13)

    plt.suptitle(f"{pol} 多维全景归因分析 (Pseudo R²={pseudo_r2:.3f}, GCV={gcv_score:.2f})", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(f'GAM_Ultimate_{pol}.png', dpi=300, bbox_inches='tight')
    plt.close()

df_gam_summary = pd.DataFrame(gam_results)
df_gam_summary.to_csv('GAM_GodMode_Summary.csv', index=False, encoding='utf-8-sig')
print("\n国一封神版 GAM 运行完毕！请查看高清 2D 交互图与 GCV 统计表。")