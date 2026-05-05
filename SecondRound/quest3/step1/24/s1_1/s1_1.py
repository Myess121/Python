# -*- coding: utf-8 -*-
"""
Q3 完整修复版：峰值增强预报系统
修复内容：
1. 定义了缺失的 train_mask 和 test_mask
2. 修复了 polyfit 趋势计算的维度问题
3. 修正了索引切片与布尔掩码的混合使用错误
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 全局配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 核心指标与辅助函数
# ==========================================
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1e-8
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100


def calculate_iaqi(C, pollutant):
    """国标分段线性插值 (HJ 633-2012)"""
    levels = [0, 50, 100, 150, 200, 300, 400, 500]
    bp_dict = {
        'PM2.5': [0, 35, 75, 115, 150, 250, 350, 500],
        'PM10': [0, 50, 150, 250, 350, 420, 500, 600],
        'O3': [0, 160, 200, 300, 400, 800, 1000, 1200],
        'NO2': [0, 100, 200, 700, 1200, 2340, 3090, 3840],
        'SO2': [0, 150, 500, 650, 800, 1600, 2100, 2620],
        'CO': [0, 5, 10, 35, 60, 90, 120, 150]
    }
    bp = bp_dict.get(pollutant, [0, 500])
    for i in range(1, len(bp)):
        if C <= bp[i]:
            return np.ceil(((levels[i] - levels[i - 1]) / (bp[i] - bp[i - 1])) * (C - bp[i - 1]) + levels[i - 1])
    return 500


def get_comprehensive_aqi(row):
    # 适配 Series 或 Dict
    return max([calculate_iaqi(row[p], p) for p in ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']])


# ==========================================
# 2. 主程序流程
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print(" Q3 峰值增强预报系统启动 (生产修复版)")
    print("=" * 60)

    # 2.1 数据加载
    print("\n📂 1. 加载数据...")
    try:
        df = pd.read_excel('Adata2.xlsx')
    except:
        print(" 错误：找不到 Adata2.xlsx")
        exit(1)

    df['datetime'] = pd.to_datetime(df['datetime'])
    pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    meteo_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']

    # 数据清洗
    for p in pollutants:
        df[p] = pd.to_numeric(df[p], errors='coerce')

    # 统一 CO 单位（假设原始为 mg/m3）
    if df['CO'].max() < 100:
        df['CO'] = df['CO'] * 1000

    df = df.dropna().reset_index(drop=True)

    # 2.2 特征工程
    print("🔧 2. 构建滑动窗口特征...")
    LOOK_BACK = 24
    HORIZON = 24
    feature_cols = pollutants + meteo_cols + ['hour_sin', 'hour_cos']

    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)

    X_list, Y_list, W_list, date_list = [], [], [], []

    valid_len = len(df) - LOOK_BACK - HORIZON
    for i in range(valid_len):
        # 输入：过去 24h
        x_window = df[feature_cols].iloc[i: i + LOOK_BACK].values.flatten()
        max_vals = df[pollutants].iloc[i: i + LOOK_BACK].max().values

        # 趋势特征：修复 polyfit 维度
        y_trend_part = df[pollutants].iloc[i + LOOK_BACK - 2: i + LOOK_BACK].values
        trend = (y_trend_part[1] - y_trend_part[0])  # 简单一阶差分代替 polyfit 避免单点错误

        X_list.append(np.concatenate([x_window, max_vals, trend]))
        Y_list.append(df[pollutants].iloc[i + LOOK_BACK: i + LOOK_BACK + HORIZON].values.flatten())

        # 权重：高污染样本加权
        future_max = df[pollutants].iloc[i + LOOK_BACK: i + LOOK_BACK + HORIZON].max()
        aqi_weight = get_comprehensive_aqi(future_max)
        W_list.append(1.0 + 2.0 * (aqi_weight / 150.0))

        # 记录该样本对应的预测起始时间
        date_list.append(df['datetime'].iloc[i + LOOK_BACK])

    X_all = np.array(X_list)
    Y_all = np.array(Y_list)
    W_all = np.array(W_list)
    dates = pd.Series(date_list)

    # 2.3 划分数据集 (修复关键点：定义 mask)
    print("🗓️ 3. 划分训练/测试集 (1-2月 vs 3月)...")
    train_mask = dates.dt.month < 3
    test_mask = dates.dt.month == 3

    # 进一步划分训练集和验证集（用于拟合校准器）
    train_indices = np.where(train_mask)[0]
    split_idx = int(len(train_indices) * 0.9)

    idx_train = train_indices[:split_idx]
    idx_val = train_indices[split_idx:]
    idx_test = np.where(test_mask)[0]

    X_train, Y_train, W_train = X_all[idx_train], Y_all[idx_train], W_all[idx_train]
    X_val, Y_val = X_all[idx_val], Y_all[idx_val]
    X_test, Y_test = X_all[idx_test], Y_all[idx_test]

    # 2.4 模型训练
    print("🧠 4. 训练多输出梯度提升树...")
    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=200, max_depth=7, random_state=42)
    )
    model.fit(X_train, Y_train, sample_weight=W_train)

    # 2.5 峰值校准
    print("⚙️ 5. 计算全局峰值校准系数...")
    Y_val_pred = model.predict(X_val)
    # 只针对高浓度区域拟合
    v_true = Y_val.flatten()
    v_pred = Y_val_pred.flatten()
    high_mask = v_true > 40

    if high_mask.sum() > 10:
        alpha, beta = np.polyfit(v_pred[high_mask], v_true[high_mask], 1)
        alpha = np.clip(alpha, 1.0, 1.5)  # 限制缩放强度
    else:
        alpha, beta = 1.0, 0.0

    # 2.6 预测与应用校准
    print("🚀 6. 执行预测并校准...")
    Y_test_raw = model.predict(X_test)
    Y_test_calib = np.maximum(0, alpha * Y_test_raw + beta)

    # 2.7 结果整合
    # 提取每个预测窗口的第一小时作为展示结果
    test_dates_out = dates[idx_test].values
    res_data = {'datetime': test_dates_out}

    for i, p in enumerate(pollutants):
        # Y_test_calib 形状是 (N, 144), 每 24 列是一个污染物
        res_data[f'true_{p}'] = Y_test[:, i]  # 取 T+1 时刻
        res_data[f'pred_{p}'] = Y_test_calib[:, i]

    df_res = pd.DataFrame(res_data)

    # 2.8 指标报表
    print("\n" + "=" * 50)
    print(f"{'污染物':<8} | {'RMSE':<8} | {'MAE':<8} | {'R²':<8}")
    print("-" * 50)
    for p in pollutants:
        rmse = np.sqrt(mean_squared_error(df_res[f'true_{p}'], df_res[f'pred_{p}']))
        mae = mean_absolute_error(df_res[f'true_{p}'], df_res[f'pred_{p}'])
        r2 = r2_score(df_res[f'true_{p}'], df_res[f'pred_{p}'])
        print(f"{p:<9} | {rmse:<8.2f} | {mae:<8.2f} | {r2:<8.3f}")
    print("=" * 50)

    # 2.9 绘图
    plt.figure(figsize=(12, 5))
    # 以 PM2.5 为例展示
    plt.plot(df_res['datetime'], df_res['true_PM2.5'], label='真实值', alpha=0.7)
    plt.plot(df_res['datetime'], df_res['pred_PM2.5'], label='校准预测', linestyle='--')
    plt.title('3月 PM2.5 预测趋势图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Q3_Result_Preview.png')

    print("\n✨ 任务完成！结果已保存至 Q3_Result_Preview.png")