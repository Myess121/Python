# -*- coding: utf-8 -*-
"""
Q3_滚动迭代对比版：22:00起点，自回归预测次日24小时
⚠️ 仅用于对比验证，实际部署请回退到直接多步策略
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 特征工程（与你rf2_n.py完全一致） =================
def build_features(df):
    df_feat = df.copy()
    df_feat['datetime'] = pd.to_datetime(df_feat['datetime'], errors='coerce')
    df_feat = df_feat.dropna(subset=['datetime'])

    pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    df_feat['CO'] = pd.to_numeric(df_feat['CO'], errors='coerce') * 1000  # 统一为 μg/m³ 方便计算

    meteo_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']
    for lag in [1, 6, 12, 13, 14]: df_feat[f'Temp_lag{lag}'] = df_feat['temperature'].shift(lag)
    for lag in [1, 13, 24]: df_feat[f'Hum_lag{lag}'] = df_feat['humidity'].shift(lag)
    for lag in [1, 6, 12]:
        for col in ['wind_speed', 'wind_direction', 'pressure']:
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)

    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['datetime'].dt.hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['datetime'].dt.hour / 24)

    for p in pollutants:
        if p != 'CO': df_feat[p] = pd.to_numeric(df_feat[p], errors='coerce')
        df_feat[f'{p}_lag1'] = df_feat[p].shift(1)
        df_feat[f'{p}_delta'] = df_feat[p] - df_feat[f'{p}_lag1']

    return df_feat.dropna().reset_index(drop=True), pollutants


# ================= 2. 主程序 =================
if __name__ == "__main__":
    print("📥 加载数据与特征工程...")
    df = pd.read_excel('Adata2.xlsx')
    df_proc, pollutants = build_features(df)

    delta_cols = [f'{p}_delta' for p in pollutants]
    feature_cols = [c for c in df_proc.columns if c not in pollutants + delta_cols + ['datetime', 'hour']]

    X = df_proc[feature_cols]
    Y_delta = df_proc[delta_cols]

    train_mask = df_proc['datetime'].dt.month <= 2
    test_mask = df_proc['datetime'].dt.month == 3

    print(" 训练 RF-ΔY 模型...")
    rf_base = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(rf_base)
    model.fit(X[train_mask], Y_delta[train_mask])

    # ================= 🔄 滚动迭代预测核心 =================
    print("🚀 启动 22:00 锚定滚动迭代预测 (次日24h)...")
    test_df = df_proc[test_mask].reset_index(drop=True)
    Y_pred_roll = pd.DataFrame(index=test_df.index, columns=pollutants)

    # 找到所有决策起点 22:00
    anchors = test_df[test_df['datetime'].dt.hour == 22].index.tolist()

    for start in anchors:
        # 起点状态：必须用当天的真实监测值
        curr_state = test_df.loc[start, pollutants].values.astype(float)

        for step in range(1, 25):  # 预测未来 1~24 小时
            idx = start + step
            if idx >= len(test_df): break

            # 构造输入行（严格对齐列名）
            row_input = pd.DataFrame([test_df.iloc[idx]], columns=feature_cols)

            # ️ 滚动核心：将 lag1 替换为当前迭代值（而非测试集原始值）
            for i, p in enumerate(pollutants):
                row_input[f'{p}_lag1'] = curr_state[i]

            # 预测变化量
            delta_pred = model.predict(row_input)[0]

            # 更新状态（物理非负约束）
            curr_state = np.maximum(0, curr_state + delta_pred)
            Y_pred_roll.iloc[idx] = curr_state

    # ================= 评估与对比 =================
    valid_idx = Y_pred_roll.dropna().index
    Y_true = test_df.loc[valid_idx, pollutants].astype(float)
    Y_pred = Y_pred_roll.loc[valid_idx].astype(float)

    print("\n 滚动迭代预测结果（3月全月汇总）")
    print("=" * 60)
    for p in pollutants:
        yt, yp = Y_true[p].values, Y_pred[p].values
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        # 简单方向准确率
        dir_acc = np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yp))) * 100
        print(f"{p: <6} | R²={r2: .3f} | MAE={mae: .1f} | 方向准确率={dir_acc: .1f}%")
    print("=" * 60)

    # 绘制对比图
    plt.figure(figsize=(14, 5))
    plt.plot(test_df.loc[valid_idx, 'datetime'], Y_true['PM2.5'], label='真实 PM2.5', color='#1f77b4', linewidth=2)
    plt.plot(test_df.loc[valid_idx, 'datetime'], Y_pred['PM2.5'], label='滚动预测 PM2.5', color='#ff7f0e',
             linestyle='--', linewidth=2)
    plt.title("22:00锚定滚动迭代预测效果（误差累积演示）", fontsize=15, fontweight='bold')
    plt.ylabel("浓度 (μg/m³)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    plt.tight_layout()
    plt.savefig('Rolling_Forecast_Comparison.png', dpi=300)
    print("✅ 对比图已保存: Rolling_Forecast_Comparison.png")