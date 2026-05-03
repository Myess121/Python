import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def build_pure_meteo_features(df):
    df_feat = df.copy()

    # 1. 时间清洗与周期编码 (题目要求的“基于时间”)
    df_feat['datetime'] = pd.to_datetime(df_feat['datetime'], errors='coerce')
    df_feat = df_feat.dropna(subset=['datetime'])
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['datetime'].dt.hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['datetime'].dt.hour / 24)

    # 2. 气象特征基础 (题目要求的“基于温湿度”，外加风压扩展)
    meteo_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']

    # 3. 精细滞后特征构建 (完全基于 Q1 Spearman 检验峰值提取)
    # 温度显著峰值: 1, 6, 12, 13, 14
    temp_lags = [1, 6, 12, 13, 14]
    for lag in temp_lags:
        df_feat[f'Temp_lag{lag}'] = df_feat['temperature'].shift(lag)

    # 湿度显著峰值: 1, 13, 24
    hum_lags = [1, 13, 24]
    for lag in hum_lags:
        df_feat[f'Hum_lag{lag}'] = df_feat['humidity'].shift(lag)

    # 动力学要素基础滞后 (风速气压等，标配 1,6,12)
    other_meteo_cols = ['wind_speed', 'wind_direction', 'pressure']
    for lag in [1, 6, 12]:
        for col in other_meteo_cols:
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)

    # 4. 污染物处理 (单位统一)
    pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    df_feat['CO'] = pd.to_numeric(df_feat['CO'], errors='coerce') * 1000
    for p in pollutants:
        if p != 'CO':
            df_feat[p] = pd.to_numeric(df_feat[p], errors='coerce')
        # ⚠️ 注意：计算滞后仅用于评估 Baseline(持久性基准)，坚决不放入模型特征集 X！
        df_feat[f'{p}_lag1'] = df_feat[p].shift(1)

    return df_feat.dropna().reset_index(drop=True), pollutants, meteo_cols, temp_lags, hum_lags


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1e-8
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100


if __name__ == "__main__":
    print("📂 开始读取数据 Adata2...")
    file_path = 'Adata2.xlsx'
    df = pd.read_excel(file_path)

    print("⚙️ 正在构建【纯气象驱动特征】 (彻底剔除污染物历史浓度)...")
    df_processed, pollutants, meteo_cols, temp_lags, hum_lags = build_pure_meteo_features(df)

    # 构建严格的纯气象特征集 X
    feature_cols = meteo_cols + ['hour_sin', 'hour_cos'] + \
                   [f'Temp_lag{l}' for l in temp_lags] + \
                   [f'Hum_lag{l}' for l in hum_lags] + \
                   [f'{c}_lag{l}' for l in [1, 6, 12] for c in ['wind_speed', 'wind_direction', 'pressure']]

    X = df_processed[feature_cols].values
    Y = df_processed[pollutants].values  # 直接预测绝对浓度 Y

    # 时序切分
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print("🏗️ 正在训练【纯气象驱动】RF 模型 (直接预测绝对浓度 Y)...")
    rf_base = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(rf_base)
    model.fit(X_train, Y_train)

    print("📊 预测完毕，正在输出阶段一评估指标...")
    Y_pred = model.predict(X_test)

    # 获取测试集的真实数据用于比对
    Y_test_real = pd.DataFrame(Y_test, columns=pollutants)
    Y_pred_real = pd.DataFrame(Y_pred, columns=pollutants)

    # 提取基准模型值 (即 Lag1，模拟只靠惯性)
    baseline_cols = [f'{p}_lag1' for p in pollutants]
    Y_base_real = df_processed[baseline_cols].iloc[split_idx:].values

    # CO 单位回归 mg/m3
    idx_co = pollutants.index('CO')
    Y_test_real.iloc[:, idx_co] /= 1000
    Y_pred_real.iloc[:, idx_co] /= 1000
    Y_base_real[:, idx_co] /= 1000

    print("\n" + "=" * 105)
    print("🏆 阶段一：纯气象驱动模型性能 (完全剥离历史污染，严格检验气象解释力)")
    print("=" * 105)
    print(
        f"{'污染物':<6} | {'纯气象 R2':<9} | {'基准(惯性) R2':<11} | {'纯气象 RMSE':<10} | {'纯气象 MAE':<9} | {'sMAPE(%)':<8}")
    print("-" * 105)

    results_data = []
    for i, p in enumerate(pollutants):
        y_t = Y_test_real[p].values
        y_p = Y_pred_real[p].values
        y_b = Y_base_real[:, i]

        r2_m = r2_score(y_t, y_p)
        r2_b = r2_score(y_t, y_b)
        rmse_m = np.sqrt(mean_squared_error(y_t, y_p))
        mae_m = mean_absolute_error(y_t, y_p)
        smape_m = smape(y_t, y_p)

        print(f"[{p:<5}] | {r2_m:<10.4f} | {r2_b:<13.4f} | {rmse_m:<11.2f} | {mae_m:<10.2f} | {smape_m:<8.2f}%")

        results_data.append({
            '污染物': p,
            '纯气象 R2': r2_m,
            '基准(惯性) R2': r2_b,
            '纯气象 RMSE': rmse_m,
            '纯气象 MAE': mae_m,
            '纯气象 sMAPE(%)': smape_m
        })
    print("=" * 105)

    pd.DataFrame(results_data).to_csv('Phase1_Pure_Meteo_Results.csv', index=False, encoding='utf-8-sig')

    # ================= 协同矩阵验证 =================
    print("\n" + "=" * 55)
    print("📊 多污染物协同网络拓扑误差 (Frobenius 范数)")
    print("=" * 55)

    corr_true = Y_test_real.corr()

    # 【核心1】提取阶段一（纯气象）的 F-dist
    corr_pred_meteo = Y_pred_real.corr()
    frob_norm_meteo = np.linalg.norm(corr_true.values - corr_pred_meteo.values)

    # 【核心2】提取阶段二（纯惯性/Lag1）的 F-dist
    corr_pred_inertia = pd.DataFrame(Y_base_real, columns=pollutants).corr()
    frob_norm_inertia = np.linalg.norm(corr_true.values - corr_pred_inertia.values)

    print(f"🎯 [阶段一] 纯气象驱动模型 F-dist : {frob_norm_meteo:.4f}")
    print(f"🎯 [阶段二] 纯惯性基准模型 F-dist : {frob_norm_inertia:.4f}")
    print("=" * 55)

    # 下面保留你原来的画图代码...


    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(corr_true, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0], fmt='.2f')
    axes[0].set_title("真实浓度相关性协同矩阵")
    sns.heatmap(corr_pred_meteo, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1], fmt='.2f')
    axes[1].set_title(f"纯气象模型预测协同矩阵\n(Frobenius 距离: {frob_norm_meteo:.4f})")
    plt.tight_layout()
    plt.savefig('Phase1_Pure_Meteo_Synergy.png', dpi=300)

    print("✅ 纯气象驱动试验运行完毕！CSV 与 热力图已保存！")