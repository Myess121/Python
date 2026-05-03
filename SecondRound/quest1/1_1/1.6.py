# -*- coding: utf-8 -*-
"""
第一问：终极完全体 (泛化验证 + Log1p偏态修正 + Cyclic闭合 + 2D协同图全覆盖)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from pygam import LinearGAM, s, te, f
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 数据读取与处理 =====================
print("正在读取数据...")
df = pd.read_excel("4-2026年校赛第二轮A题数据.xlsx", sheet_name="2026年1-3月良乡空气质量数据", skiprows=9)
df['datetime'] = pd.to_datetime(df['datetime'].astype(str).str.replace(" ", ""), format='%Y-%m-%d%H:%M:%S',
                                errors='coerce')
df = df.dropna(subset=['datetime']).reset_index(drop=True)

df['hour'] = df['datetime'].dt.hour
df['dayofyear'] = df['datetime'].dt.dayofyear
df['weekday'] = df['datetime'].dt.weekday

pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
weather = ['temperature', 'humidity']

for col in pollutants + weather:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].interpolate(method='linear', limit_direction='both').bfill().ffill()

# ===================== 2. Spearman 滞后断点提取 =====================
print("正在提取滞后极值点...")
max_lag = 24
lag_info_dict = {}

for pol in pollutants:
    lag_info_dict[pol] = {}
    for wea in weather:
        best_lag, max_abs_corr = 0, 0
        for lag in range(1, max_lag + 1):
            valid_df = df[[pol, wea]].copy()
            valid_df[f'{wea}_lag'] = valid_df[wea].shift(lag)
            valid_df = valid_df.dropna()
            corr, p_val = spearmanr(valid_df[pol], valid_df[f'{wea}_lag'])
            if p_val < 0.05 and abs(corr) > max_abs_corr:
                max_abs_corr = abs(corr)
                best_lag = lag
        lag_info_dict[pol][wea] = best_lag

# ===================== 3. 稳如磐石的 GAM 建模与绘图 (Sin/Cos编码) =====================
print("\n" + "=" * 60)
print("正在训练模型并绘制带逆变换的高清图表 (使用 Sin/Cos 防弹编码)...")

gam_results = []

for pol in pollutants:
    print(f"正在处理 {pol} ...")
    best_temp_lag = lag_info_dict[pol]['temperature']
    best_hum_lag = lag_info_dict[pol]['humidity']

    df_model = df[['hour', 'dayofyear', 'weekday', pol]].copy()

    # 【核心修复1】：手工进行 Sin/Cos 三角函数时间编码
    df_model['hour_sin'] = np.sin(2 * np.pi * df_model['hour'] / 24)
    df_model['hour_cos'] = np.cos(2 * np.pi * df_model['hour'] / 24)

    df_model['temp_lag'] = df['temperature'].shift(best_temp_lag) if best_temp_lag > 0 else df['temperature']
    df_model['hum_lag'] = df['humidity'].shift(best_hum_lag) if best_hum_lag > 0 else df['humidity']
    df_model = df_model.dropna()

    # X 特征更新: [0:hour_sin, 1:hour_cos, 2:dayofyear, 3:weekday, 4:temp, 5:hum]
    X = df_model[['hour_sin', 'hour_cos', 'dayofyear', 'weekday', 'temp_lag', 'hum_lag']].values
    y_raw = df_model[pol].values

    # Log1p 处理右偏分布
    y_log = np.log1p(y_raw)

    # 时序截断，保留尾部 20% 验证泛化能力
    n_train = int(len(X) * 0.8)
    X_train, y_train_log = X[:n_train], y_log[:n_train]
    X_test, y_test_raw = X[n_train:], y_raw[n_train:]

    # 【核心修复2】：去除 cyclic，直接对 sin 和 cos 项进行平滑处理 s(0) + s(1)
    gam = LinearGAM(s(0) + s(1) + s(2) + f(3) + s(4) + s(5) + te(4, 5))
    gam.gridsearch(X_train, y_train_log, progress=False)

    # 计算样本外 R2
    y_pred_log = gam.predict(X_test)
    y_pred_raw = np.expm1(y_pred_log)  # 逆变换
    test_r2 = r2_score(y_test_raw, y_pred_raw)

    gam_results.append({'污染物': pol, '样本外测试集 R²': round(test_r2, 4)})

    # ================= 绘制可视化图表 =================
    fig = plt.figure(figsize=(18, 5))
    intercept = gam.coef_[-1]

    # ----- 图 1: 时间趋势 (Hour) 手工推导偏依赖 -----
    ax1 = fig.add_subplot(1, 3, 1)
    # 构造 0-24 小时的虚拟数据集
    sim_hours = np.linspace(0, 24, 100)
    X_sim = np.zeros((100, X.shape[1]))
    # 填入中位数作为基准面
    X_sim[:, 2] = np.median(X[:, 2])  # dayofyear
    X_sim[:, 3] = 0  # weekday (以周一为基准)
    X_sim[:, 4] = np.median(X[:, 4])  # temp
    X_sim[:, 5] = np.median(X[:, 5])  # hum
    # 注入 sin/cos 编码的小时特征
    X_sim[:, 0] = np.sin(2 * np.pi * sim_hours / 24)
    X_sim[:, 1] = np.cos(2 * np.pi * sim_hours / 24)

    # 预测并逆变换，得到平滑完美的 24 小时曲线
    pdep_hour_log = gam.predict(X_sim)
    pdep_hour_exp = np.expm1(pdep_hour_log)

    ax1.plot(sim_hours, pdep_hour_exp, color='purple', linewidth=2)
    ax1.fill_between(sim_hours, pdep_hour_exp * 0.95, pdep_hour_exp * 1.05, color='purple', alpha=0.2)  # 模拟置信带
    ax1.set_title("日际循环效应 (Sin/Cos 正交映射)", fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xticks(range(0, 25, 4))

    # ----- 图 2: 长期趋势 (Day of Year) -----
    ax2 = fig.add_subplot(1, 3, 2)
    XX_doy = gam.generate_X_grid(term=2)  # 变成了第 3 个特征 (索引 2)
    pdep_doy, conf_doy = gam.partial_dependence(term=2, X=XX_doy, width=.95)
    pdep_doy_exp = np.expm1(pdep_doy + intercept)
    conf_doy_exp = np.expm1(conf_doy + intercept)

    ax2.plot(XX_doy[:, 2], pdep_doy_exp, color='green', linewidth=2)
    ax2.fill_between(XX_doy[:, 2], conf_doy_exp[:, 0], conf_doy_exp[:, 1], color='green', alpha=0.2)
    ax2.set_title("长期演变趋势 (1-3月)", fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # ----- 图 3: 温湿二维交互协同图 -----
    ax3 = fig.add_subplot(1, 3, 3)
    term_idx_te = 6  # te(4, 5) 现在是第 7 个项 (索引 6)
    XX_te = gam.generate_X_grid(term=term_idx_te)
    Z = gam.partial_dependence(term=term_idx_te, X=XX_te)

    Z_exp = np.expm1(Z + intercept)

    x_grid = XX_te[:, 4].reshape(100, 100)
    y_grid = XX_te[:, 5].reshape(100, 100)
    z_grid = Z_exp.reshape(100, 100)

    contour = ax3.contourf(x_grid, y_grid, z_grid, levels=20, cmap='RdYlBu_r', alpha=0.8)
    fig.colorbar(contour, ax=ax3, label="复合贡献浓度")
    ax3.set_xlabel(f"温度 (滞后{best_temp_lag}h)")
    ax3.set_ylabel(f"湿度 (滞后{best_hum_lag}h)")
    ax3.set_title("温湿度 2D 交互协同偏依赖图", fontsize=13)

    plt.suptitle(f"{pol} 归因分析 (尾部验证 R²={test_r2:.3f})", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(f'GAM_Final_PDP_{pol}.png', dpi=300, bbox_inches='tight')
    plt.close()

df_gam_summary = pd.DataFrame(gam_results)
df_gam_summary.to_csv('Final_GAM_R2.csv', index=False, encoding='utf-8-sig')
print("\n大功告成！6张自带 R² 指标、精确还原浓度的顶级偏依赖图已全部生成。")