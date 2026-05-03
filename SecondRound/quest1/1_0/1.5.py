# -*- coding: utf-8 -*-
"""
第一问：究极无破绽版 (时序尾部验证 + Log1p偏态修正 + Sin/Cos周期闭合)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from pygam import LinearGAM, s, te, l, f
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 辅助函数：格式化断点滞后 =====================
def format_lags(lags):
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


# ===================== 1. 数据读取与【周期时间】提取 =====================
print("正在读取数据...")
df = pd.read_excel("4-2026年校赛第二轮A题数据.xlsx", sheet_name="2026年1-3月良乡空气质量数据", skiprows=9)
df['datetime'] = pd.to_datetime(df['datetime'].astype(str).str.replace(" ", ""), format='%Y-%m-%d%H:%M:%S',
                                errors='coerce')
df = df.dropna(subset=['datetime']).reset_index(drop=True)

df['hour'] = df['datetime'].dt.hour
df['dayofyear'] = df['datetime'].dt.dayofyear
df['weekday'] = df['datetime'].dt.weekday

# 【硬伤3修复】：将线性 hour 转化为圆周上的坐标 (sin/cos 编码，完美解决 0点-23点 断层)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
weather = ['temperature', 'humidity']

for col in pollutants + weather:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].interpolate(method='linear', limit_direction='both').bfill().ffill()

# ===================== 2. Spearman 离散滞后分析 (保持不变) =====================
print("正在提取滞后断点...")
max_lag = 24
lag_info_dict = {}

for pol in pollutants:
    lag_info_dict[pol] = {}
    for wea in weather:
        best_lag, max_abs_corr = 0, 0
        significant_lags = []
        for lag in range(1, max_lag + 1):
            valid_df = df[[pol, wea]].copy()
            valid_df[f'{wea}_lag'] = valid_df[wea].shift(lag)
            valid_df = valid_df.dropna()
            corr, p_val = spearmanr(valid_df[pol], valid_df[f'{wea}_lag'])
            if p_val < 0.05:
                significant_lags.append(lag)
                if abs(corr) > max_abs_corr:
                    max_abs_corr = abs(corr)
                    best_lag = lag
        lag_info_dict[pol][wea] = best_lag

# ===================== 3. 究极 GAM 建模 (Log1p + 时序切分) =====================
print("\n" + "=" * 60)
print("正在构建具备强泛化能力的 GAM 模型...")

gam_results = []

for pol in pollutants:
    best_temp_lag = lag_info_dict[pol]['temperature']
    best_hum_lag = lag_info_dict[pol]['humidity']

    df_model = df[['hour_sin', 'hour_cos', 'dayofyear', 'weekday', pol]].copy()
    df_model['temp_lag'] = df['temperature'].shift(best_temp_lag) if best_temp_lag > 0 else df['temperature']
    df_model['hum_lag'] = df['humidity'].shift(best_hum_lag) if best_hum_lag > 0 else df['humidity']
    df_model = df_model.dropna()

    # 特征: [0:hour_sin, 1:hour_cos, 2:dayofyear, 3:weekday, 4:temp, 5:hum]
    X = df_model[['hour_sin', 'hour_cos', 'dayofyear', 'weekday', 'temp_lag', 'hum_lag']].values

    # 【硬伤2修复】：处理右偏分布，使用 log1p 转换目标变量
    y_raw = df_model[pol].values
    y_log = np.log1p(y_raw)

    # 【硬伤1修复】：时序尾部切分验证，绝不打乱时间顺序
    n_train = int(len(X) * 0.8)  # 前 80% 训练，后 20% 测试
    X_train, y_train_log = X[:n_train], y_log[:n_train]
    X_test, y_test_raw = X[n_train:], y_raw[n_train:]  # 测试集保留真实值用于算R2

    # 构建模型：此时 hour 变成了 sin/cos，使用线性项 l() 即可刻画圆周运动
    gam = LinearGAM(l(0) + l(1) + s(2) + f(3) + s(4) + s(5) + te(4, 5))
    gam.gridsearch(X_train, y_train_log, progress=False)

    # 预测并逆变换还原真实浓度
    y_pred_log = gam.predict(X_test)
    y_pred_raw = np.expm1(y_pred_log)

    # 计算真实世界的样本外 R2
    test_r2 = r2_score(y_test_raw, y_pred_raw)

    gam_results.append({
        '污染物': pol,
        '外推测试集 R²': round(test_r2, 4)
    })
    print(f"{pol} 模型拟合完成 -> 样本外测试集 R²: {test_r2:.4f}")

df_gam_summary = pd.DataFrame(gam_results)
df_gam_summary.to_csv('Ultimate_GAM_Test_R2.csv', index=False, encoding='utf-8-sig')
print("\n究极版运行完毕！真实预测能力已保存。")