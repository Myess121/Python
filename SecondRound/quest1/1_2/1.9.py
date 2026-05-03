# -*- coding: utf-8 -*-
"""
第一问·终极封神版：全面统计显著性 + 自动文案生成 + 真实置信带修正
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from pygam import LinearGAM, s, te, f
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========================= 1. 数据读取与预处理 =========================
print("1. 读取并清洗数据...")
df = pd.read_excel("4-2026年校赛第二轮A题数据.xlsx",
                   sheet_name="2026年1-3月良乡空气质量数据", skiprows=9)
df['datetime'] = pd.to_datetime(
    df['datetime'].astype(str).str.replace(" ", ""),
    format='%Y-%m-%d%H:%M:%S',
    errors='coerce'
)
df = df.dropna(subset=['datetime']).reset_index(drop=True)

df['hour'] = df['datetime'].dt.hour
df['dayofyear'] = df['datetime'].dt.dayofyear
df['weekday'] = df['datetime'].dt.weekday

pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
weather = ['temperature', 'humidity']

for col in pollutants + weather:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].interpolate(method='linear', limit_direction='both').bfill().ffill()

# ========================= 2. Spearman 滞后分析 =========================
print("2. Spearman 滞后分析（提取显著窗口与峰值）...")
max_lag = 24
lag_results = []
lag_best = {p: {} for p in pollutants}

for pol in pollutants:
    for wea in weather:
        sig_lags = []
        best_lag, best_corr = 0, 0
        for lag in range(1, max_lag + 1):
            tmp = df[[pol, wea]].copy()
            tmp[f'{wea}_lag'] = tmp[wea].shift(lag)
            tmp = tmp.dropna()
            corr, pval = spearmanr(tmp[pol], tmp[f'{wea}_lag'])
            if pval < 0.05:
                sig_lags.append(lag)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

        if not sig_lags:
            window_str, best_lag, best_corr = "无显著滞后", 0, 0.0
        else:
            sig_lags_sorted = sorted(sig_lags)
            segments = []
            start = sig_lags_sorted[0]
            for i in range(1, len(sig_lags_sorted)):
                if sig_lags_sorted[i] != sig_lags_sorted[i - 1] + 1:
                    segments.append(
                        f"{start}-{sig_lags_sorted[i - 1]}h" if start != sig_lags_sorted[i - 1] else f"{start}h")
                    start = sig_lags_sorted[i]
            segments.append(f"{start}-{sig_lags_sorted[-1]}h" if start != sig_lags_sorted[-1] else f"{start}h")
            window_str = ", ".join(segments)

        lag_results.append({
            '污染物': pol, '气象要素': '温度' if wea == 'temperature' else '湿度',
            '显著滞后窗口': window_str, '最强滞后(h)': best_lag, '峰Spearman系数': round(best_corr, 4)
        })
        lag_best[pol][wea] = best_lag

df_lag = pd.DataFrame(lag_results)
df_lag.to_csv('Q1_Spearman_Lag.csv', index=False, encoding='utf-8-sig')

# --- 【硬核补丁2】：自动生成论文级别文字摘要 ---
report_lines = ["【自动生成的论文滞后分析摘要段落】\n"]
for pol in pollutants:
    temp_info = next(item for item in lag_results if item['污染物'] == pol and item['气象要素'] == '温度')
    hum_info = next(item for item in lag_results if item['污染物'] == pol and item['气象要素'] == '湿度')
    report_lines.append(
        f"经分布滞后检验，{pol} 对温度的显著响应窗口为 {temp_info['显著滞后窗口']}，"
        f"其效应在滞后 {temp_info['最强滞后(h)']} 小时达到峰值（Spearman={temp_info['峰Spearman系数']}）；"
        f"对湿度的显著影响期则集中在 {hum_info['显著滞后窗口']}，"
        f"最强反馈发生于滞后 {hum_info['最强滞后(h)']} 小时（Spearman={hum_info['峰Spearman系数']}）。"
    )
with open('Q1_Auto_Report.txt', 'w', encoding='utf-8') as txt_file:
    txt_file.write("\n".join(report_lines))

# ========================= 3. GAM 建模与全维统计显著性 =========================
print("3. 构建 GAM 模型并提取 P值/EDF 统计显著性表...")

model_metrics = []
model_details = []

for pol in pollutants:
    bt = lag_best[pol]['temperature']
    bh = lag_best[pol]['humidity']

    df_mod = df[['hour', 'dayofyear', 'weekday', pol]].copy()
    df_mod['temp_lag'] = df['temperature'].shift(bt) if bt > 0 else df['temperature']
    df_mod['hum_lag'] = df['humidity'].shift(bh) if bh > 0 else df['humidity']
    df_mod = df_mod.dropna().reset_index(drop=True)

    df_mod['hour_sin'] = np.sin(2 * np.pi * df_mod['hour'] / 24)
    df_mod['hour_cos'] = np.cos(2 * np.pi * df_mod['hour'] / 24)

    # 特征: 0:hour_sin, 1:hour_cos, 2:doy, 3:weekday, 4:temp, 5:hum
    X = df_mod[['hour_sin', 'hour_cos', 'dayofyear', 'weekday', 'temp_lag', 'hum_lag']].values
    y_raw = df_mod[pol].values
    y_log = np.log1p(y_raw)

    n_train = int(len(X) * 0.8)
    X_train, y_train_log = X[:n_train], y_log[:n_train]
    X_test, y_test_raw = X[n_train:], y_raw[n_train:]

    gam = LinearGAM(s(0) + s(1) + s(2) + f(3) + s(4) + s(5) + te(4, 5))
    gam.gridsearch(X_train, y_train_log, progress=False)

    pred_log = gam.predict(X_test)
    pred_raw = np.expm1(pred_log)
    r2_test = r2_score(y_test_raw, pred_raw)

    model_metrics.append({'污染物': pol, '样本外 R²': round(r2_test, 4)})

    # --- 【硬核补丁1】：提取真实的统计显著性 P值 和 有效自由度 EDF ---
    try:
        p_vals = gam.statistics_['p_values']
        edfs = gam.statistics_['edof_per_coef']
        term_names = ['日循环(Hour_Sin)', '日循环(Hour_Cos)', '季节趋势(DayOfYear)', '周末效应(Weekday)', '温度(Temp)',
                      '湿度(Hum)', '温湿协同交互(Te)']

        # 规避 pygam 展开大量基函数的问题，直接取每个 term 的首个主控 p 值/edf
        # 这里为了稳定性和安全性，进行了安全的索引处理
        term_count = min(len(term_names), len(p_vals) - 1)  # -1 是避开 intercept
        for i in range(term_count):
            p = p_vals[i]
            edf = edfs[i]
            sig_stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            model_details.append({
                '污染物': pol,
                '特征变量': term_names[i],
                '有效自由度(EDF)': round(edf, 3),
                'p值': format(p, '.4e'),
                '显著性(0.05)': sig_stars
            })
    except Exception as e:
        print(f"提取 {pol} 统计特性时出错（可忽略，不影响绘图）: {e}")

df_metrics = pd.DataFrame(model_metrics)
df_metrics.to_csv('Q1_GAM_Metrics.csv', index=False, encoding='utf-8-sig')

df_signif = pd.DataFrame(model_details)
df_signif.to_csv('Q1_GAM_Significance.csv', index=False, encoding='utf-8-sig')

# ========================= 4. 完美可视化（严谨置信区间） =========================
print("4. 生成偏依赖图（采用真实的 Confidence Intervals）...")

for pol in pollutants:
    bt = lag_best[pol]['temperature']
    bh = lag_best[pol]['humidity']
    df_mod = df.copy()
    df_mod['temp_lag'] = df['temperature'].shift(bt) if bt > 0 else df['temperature']
    df_mod['hum_lag'] = df['humidity'].shift(bh) if bh > 0 else df['humidity']
    df_mod = df_mod.dropna().reset_index(drop=True)

    df_mod['hour_sin'] = np.sin(2 * np.pi * df_mod['hour'] / 24)
    df_mod['hour_cos'] = np.cos(2 * np.pi * df_mod['hour'] / 24)
    X_full = df_mod[['hour_sin', 'hour_cos', 'dayofyear', 'weekday', 'temp_lag', 'hum_lag']].values
    y_full_log = np.log1p(df_mod[pol].values)

    gam_full = LinearGAM(s(0) + s(1) + s(2) + f(3) + s(4) + s(5) + te(4, 5))
    gam_full.gridsearch(X_full, y_full_log, progress=False)

    baseline = np.median(X_full, axis=0)
    baseline[3] = 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 图1：小时效应 (改为真实的 confidence_intervals) ----
    hours = np.linspace(0, 24, 100)
    X_hour = np.tile(baseline, (100, 1))
    X_hour[:, 0] = np.sin(2 * np.pi * hours / 24)
    X_hour[:, 1] = np.cos(2 * np.pi * hours / 24)

    pred_hour = gam_full.predict(X_hour)
    # 【硬核补丁3】：用 confidence_intervals 替代 prediction_intervals
    conf_hour = gam_full.confidence_intervals(X_hour, width=0.95)

    pred_hour_exp = np.expm1(pred_hour)
    conf_hour_exp = np.expm1(conf_hour)

    axes[0].plot(hours, pred_hour_exp, color='purple', lw=2)
    axes[0].fill_between(hours, conf_hour_exp[:, 0], conf_hour_exp[:, 1], color='purple', alpha=0.3)
    axes[0].set_title("日际循环效应 (95% 置信带)", fontsize=13)
    axes[0].set_xlabel("小时")
    axes[0].set_ylabel(f"{pol} 浓度")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_xticks(range(0, 25, 4))

    # ---- 图2：长期趋势 ----
    doys = np.linspace(1, 90, 100)
    X_doy = np.tile(baseline, (100, 1))
    X_doy[:, 2] = doys

    pred_doy = gam_full.predict(X_doy)
    conf_doy = gam_full.confidence_intervals(X_doy, width=0.95)

    pred_doy_exp = np.expm1(pred_doy)
    conf_doy_exp = np.expm1(conf_doy)

    axes[1].plot(doys, pred_doy_exp, color='green', lw=2)
    axes[1].fill_between(doys, conf_doy_exp[:, 0], conf_doy_exp[:, 1], color='green', alpha=0.3)
    axes[1].set_title("季节演变趋势 (1-3月)", fontsize=13)
    axes[1].set_xlabel("年内第几天")
    axes[1].set_ylabel(f"{pol} 浓度")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # ---- 图3：温湿协同曲面 ----
    temp_grid = np.linspace(df_mod['temp_lag'].min(), df_mod['temp_lag'].max(), 50)
    hum_grid = np.linspace(df_mod['hum_lag'].min(), df_mod['hum_lag'].max(), 50)
    Temp, Hum = np.meshgrid(temp_grid, hum_grid)
    grid_pts = np.c_[Temp.ravel(), Hum.ravel()]
    X_surf = np.tile(baseline, (len(grid_pts), 1))
    X_surf[:, 4] = grid_pts[:, 0]
    X_surf[:, 5] = grid_pts[:, 1]

    pred_surf = gam_full.predict(X_surf)
    pred_surf_exp = np.expm1(pred_surf).reshape(Temp.shape)

    cs = axes[2].contourf(Temp, Hum, pred_surf_exp, levels=20, cmap='RdYlBu_r', alpha=0.8)
    plt.colorbar(cs, ax=axes[2], label=f"{pol} 浓度")
    axes[2].set_xlabel(f"温度 (滞后{bt}h)")
    axes[2].set_ylabel(f"湿度 (滞后{bh}h)")
    axes[2].set_title("温湿2D交互协同偏依赖图", fontsize=13)

    current_r2 = next(item['样本外 R²'] for item in model_metrics if item['污染物'] == pol)
    plt.suptitle(f"{pol} 归因分析 | 样本外 R² = {current_r2:.3f}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'Q1_PDP_Final_{pol}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\n✅ 究极封神版运行完毕！所有表格、报告、高清图表已生成。可以收工转战第二问了！")

# ========================= 4.5 进阶可视化：分面对比图（清晰、严谨、不杂乱） =========================
print("正在生成六种污染物分面对比图...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, pol in enumerate(pollutants):
    # 获取数据（逻辑同前，确保获取了平滑后的 y_h 序列）
    bt, bh = lag_best[pol]['temperature'], lag_best[pol]['humidity']
    df_mod = df.copy()
    df_mod['temp_lag'] = df['temperature'].shift(bt) if bt > 0 else df['temperature']
    df_mod['hum_lag'] = df['humidity'].shift(bh) if bh > 0 else df['humidity']
    df_mod = df_mod.dropna().reset_index(drop=True)
    df_mod['hour_sin'] = np.sin(2 * np.pi * df_mod['hour'] / 24)
    df_mod['hour_cos'] = np.cos(2 * np.pi * df_mod['hour'] / 24)
    X_f = df_mod[['hour_sin', 'hour_cos', 'dayofyear', 'weekday', 'temp_lag', 'hum_lag']].values
    gam_p = LinearGAM(s(0) + s(1) + s(2) + f(3) + s(4) + s(5) + te(4, 5)).fit(X_f, np.log1p(df_mod[pol].values))

    # 模拟 24 小时数据
    h_axis = np.linspace(0, 24, 100)
    X_plot = np.tile(np.median(X_f, axis=0), (100, 1))
    X_plot[:, 0], X_plot[:, 1] = np.sin(2 * np.pi * h_axis / 24), np.cos(2 * np.pi * h_axis / 24)
    y_plot = np.expm1(gam_p.predict(X_plot))
    y_norm = (y_plot - y_plot.min()) / (y_plot.max() - y_plot.min())  # 标准化

    # 在对应的子图中绘图
    axes[i].plot(h_axis, y_norm, color=colors[i], lw=3)
    axes[i].fill_between(h_axis, 0, y_norm, color=colors[i], alpha=0.1)  # 填充阴影增加丰满感
    axes[i].set_title(f"{pol} 日变化特征", fontsize=14, fontweight='bold')
    axes[i].grid(True, linestyle='--', alpha=0.4)
    if i >= 3: axes[i].set_xlabel("小时")
    if i % 3 == 0: axes[i].set_ylabel("标准化浓度")

plt.suptitle("六种污染物日际循环特征分面对比图 (标准化)", fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig('Q1_Pollutants_Facet_Plot.png', dpi=300, bbox_inches='tight')
plt.show()
# ========================= 4.6 补充产出：长期趋势分面对比图 (1-3月演变) =========================
print("正在生成六种污染物【长期趋势】分面对比图...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, pol in enumerate(pollutants):
    bt, bh = lag_best[pol]['temperature'], lag_best[pol]['humidity']
    df_mod = df.copy()
    df_mod['temp_lag'] = df['temperature'].shift(bt) if bt > 0 else df['temperature']
    df_mod['hum_lag'] = df['humidity'].shift(bh) if bh > 0 else df['humidity']
    df_mod = df_mod.dropna().reset_index(drop=True)
    df_mod['hour_sin'] = np.sin(2 * np.pi * df_mod['hour'] / 24)
    df_mod['hour_cos'] = np.cos(2 * np.pi * df_mod['hour'] / 24)
    X_f = df_mod[['hour_sin', 'hour_cos', 'dayofyear', 'weekday', 'temp_lag', 'hum_lag']].values

    # 拟合全量 GAM 模型
    gam_p = LinearGAM(s(0) + s(1) + s(2) + f(3) + s(4) + s(5) + te(4, 5)).fit(X_f, np.log1p(df_mod[pol].values))

    # 模拟 1-90 天的数据（长期趋势）
    doy_axis = np.linspace(1, 90, 100)
    X_plot = np.tile(np.median(X_f, axis=0), (100, 1))
    X_plot[:, 2] = doy_axis  # 改变 Day of Year 特征

    y_plot = np.expm1(gam_p.predict(X_plot))
    y_norm = (y_plot - y_plot.min()) / (y_plot.max() - y_plot.min())  # 标准化

    # 在对应的子图中绘图
    axes[i].plot(doy_axis, y_norm, color=colors[i], lw=3)
    axes[i].fill_between(doy_axis, 0, y_norm, color=colors[i], alpha=0.15)
    axes[i].set_title(f"{pol} 季节演变趋势", fontsize=14, fontweight='bold')
    axes[i].grid(True, linestyle='--', alpha=0.4)
    if i >= 3: axes[i].set_xlabel("年内第几天 (1-90)")
    if i % 3 == 0: axes[i].set_ylabel("标准化浓度")

plt.suptitle("六种污染物长期演变趋势分面对比图 (1-3月)", fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig('Q1_Pollutants_Trend_Facet.png', dpi=300, bbox_inches='tight')
plt.show()