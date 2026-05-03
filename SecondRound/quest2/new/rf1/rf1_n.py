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


def build_features(df):
    df_feat = df.copy()

    # 1. 统一脏数据清洗 (针对新数据格式)
    df_feat['datetime'] = pd.to_datetime(df_feat['datetime'], errors='coerce')
    df_feat = df_feat.dropna(subset=['datetime'])

    # 🌟 2. 气象特征全面升级 (加入风速、风向、气压)
    meteo_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']

    # 构建气象滞后特征
    # 保持原有温湿度的精细滞后结论，新增要素统一取 1, 6, 12 小时滞后
    temp_lags = [1, 6, 12, 13, 14]
    for lag in temp_lags:
        df_feat[f'Temp_lag{lag}'] = df_feat['temperature'].shift(lag)

    hum_lags = [1, 13, 24]
    for lag in hum_lags:
        df_feat[f'Hum_lag{lag}'] = df_feat['humidity'].shift(lag)

    new_meteo_cols = ['wind_speed', 'wind_direction', 'pressure']
    for lag in [1, 6, 12]:
        for col in new_meteo_cols:
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)

    # 时间周期编码
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['datetime'].dt.hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['datetime'].dt.hour / 24)

    # 3. 污染物差分处理
    pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    df_feat['CO'] = pd.to_numeric(df_feat['CO'], errors='coerce') * 1000

    for p in pollutants:
        if p != 'CO':
            df_feat[p] = pd.to_numeric(df_feat[p], errors='coerce')
        # 构造滞后 1 小时特征
        df_feat[f'{p}_lag1'] = df_feat[p].shift(1)
        # 构造目标变量 ΔY
        df_feat[f'{p}_delta'] = df_feat[p] - df_feat[f'{p}_lag1']

    return df_feat.dropna().reset_index(drop=True), pollutants, meteo_cols


if __name__ == "__main__":
    print("开始读取新版数据 Adata2...")
    # 💡 组长注意：根据实际文件格式调整后缀 .xlsx 或 .csv
    file_path = 'Adata2.xlsx'
    df = pd.read_excel(file_path)  # 新数据不需要 skiprows=9

    print("正在进行特征工程 (包含 ΔY 差分与全量气象特征)...")
    df_processed, pollutants, meteo_cols = build_features(df)

    # 3. 特征矩阵构建
    delta_cols = [f'{p}_delta' for p in pollutants]

    # 剔除目标列、原始污染物列、时间列，保留天气和滞后项
    # 🔑 修正：必须保留所有当期气象要素
    drop_cols = pollutants + delta_cols + ['datetime', 'hour']
    X = df_processed.drop(columns=[c for c in drop_cols if c in df_processed.columns])
    Y_delta = df_processed[delta_cols]

    # 时序切分
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_delta_train, Y_delta_test = Y_delta.iloc[:split_idx], Y_delta.iloc[split_idx:]

    print("正在训练基于差分与多维气象的协同随机森林模型...")
    rf_base = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(rf_base)
    model.fit(X_train, Y_delta_train)

    print("预测完毕，正在还原真实浓度与真实量纲...")
    Y_delta_pred = model.predict(X_test)
    Y_delta_pred_df = pd.DataFrame(Y_delta_pred, columns=delta_cols, index=X_test.index)

    Y_pred_real = pd.DataFrame(index=X_test.index)
    Y_test_real = df_processed[pollutants].loc[X_test.index].copy()

    for p in pollutants:
        # 还原：Lag1 + ΔY_Pred
        Y_pred_real[p] = X_test[f'{p}_lag1'] + Y_delta_pred_df[f'{p}_delta']

    # 还原 CO 单位
    Y_pred_real['CO'] /= 1000
    Y_test_real['CO'] /= 1000

    # ---------------- 4. 指标输出 ----------------
    print("\n" + "=" * 80)
    print("🏆 升级版模型竞技场 (基准 Baseline VS 全气象 RF)")
    print("=" * 80)

    for p in pollutants:
        r2_model = r2_score(Y_test_real[p], Y_pred_real[p])
        y_base = X_test[f'{p}_lag1'].values / 1000 if p == 'CO' else X_test[f'{p}_lag1'].values
        r2_base = r2_score(Y_test_real[p], y_base)
        rmse_model = np.sqrt(mean_squared_error(Y_test_real[p], Y_pred_real[p]))
        rmse_base = np.sqrt(mean_squared_error(Y_test_real[p], y_base))
        rmse_imp = (rmse_base - rmse_model) / rmse_base * 100

        print(
            f"[{p:<5}] | 本文 R2: {r2_model:.3f} (基准: {r2_base:.3f}) | 本文 RMSE: {rmse_model:<6.2f} (基准: {rmse_base:<6.2f}) -> 误差降低: {rmse_imp:.1f}%")

    # ---------------- 5. 协同验证图 ----------------
    corr_true = Y_test_real.corr()
    corr_pred = Y_pred_real.corr()
    frob_norm = np.linalg.norm(corr_true.values - corr_pred.values)
    print(f"\n【协同矩阵指标】绝对浓度 Frobenius 范数距离: {frob_norm:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(corr_true, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0], fmt='.2f')
    axes[0].set_title("真实浓度相关性矩阵")
    sns.heatmap(corr_pred, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1], fmt='.2f')
    axes[1].set_title(f"模型预测相关性矩阵 (F-dist: {frob_norm:.4f})")
    plt.tight_layout()
    plt.savefig('Q2_Synergy_Correlation_V2.png', dpi=300)

    # ΔY 协同
    delta_corr_true = Y_delta_test.corr()
    delta_corr_pred = Y_delta_pred_df.corr()
    frob_delta = np.linalg.norm(delta_corr_true.values - delta_corr_pred.values)
    print(f"【协同矩阵指标】动态变化(ΔY) Frobenius 范数距离: {frob_delta:.4f}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(delta_corr_true, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes2[0], fmt='.2f',
                xticklabels=pollutants, yticklabels=pollutants)
    axes2[0].set_title("真实浓度变化量(ΔY)协同矩阵")
    sns.heatmap(delta_corr_pred, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes2[1], fmt='.2f',
                xticklabels=pollutants, yticklabels=pollutants)
    axes2[1].set_title("模型预测变化量(ΔY)协同矩阵")
    plt.tight_layout()
    plt.savefig('Q2_Synergy_Delta_Correlation_V2.png', dpi=300)

    # ---------------- 6. 结果存表 ----------------
    results_data = []
    for i, p in enumerate(pollutants):
        y_t = Y_test_real.iloc[:, i]
        y_p = Y_pred_real.iloc[:, i]
        y_b = X_test[f'{p}_lag1'].values / 1000 if p == 'CO' else X_test[f'{p}_lag1'].values

        results_data.append({
            '污染物': p,
            '本文 R2': r2_score(y_t, y_p),
            '基准 R2': r2_score(y_t, y_b),
            '本文 RMSE': np.sqrt(mean_squared_error(y_t, y_p)),
            '基准 RMSE': np.sqrt(mean_squared_error(y_t, y_b)),
            '误差降低率(%)': ((np.sqrt(mean_squared_error(y_t, y_b)) - np.sqrt(mean_squared_error(y_t, y_p))) / np.sqrt(
                mean_squared_error(y_t, y_b))) * 100
        })

    df_final_results = pd.DataFrame(results_data)
    df_final_results.to_csv('RF_Model_Results_V2.csv', index=False, encoding='utf-8-sig')
    print("\n📊 结果已保存至 'RF_Model_Results_V2.csv'")