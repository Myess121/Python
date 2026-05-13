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
        df_feat[f'{p}_lag1'] = df_feat[p].shift(1)
        df_feat[f'{p}_delta'] = df_feat[p] - df_feat[f'{p}_lag1']

    return df_feat.dropna().reset_index(drop=True), pollutants, meteo_cols


# 🌟 新增：SMAPE 误差计算函数 (消除量纲差异)
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1e-8
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100


if __name__ == "__main__":
    print("开始读取新版数据 Adata2...")
    file_path = 'Adata2.xlsx'
    df = pd.read_excel(file_path)

    print("正在进行特征工程 (包含 ΔY 差分与全量气象特征)...")
    df_processed, pollutants, meteo_cols = build_features(df)

    # 3. 特征矩阵构建
    delta_cols = [f'{p}_delta' for p in pollutants]
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

    # 🔧 修复 1：安全对齐隐患，直接用 iloc 长度切片，100% 对应测试集
    Y_test_real = df_processed[pollutants].iloc[split_idx:].copy()
    Y_pred_real = pd.DataFrame(index=Y_test_real.index)

    for p in pollutants:
        # 🔧 修复 2：还原公式加 .values，脱离 Pandas 索引绑定，防止错位
        Y_pred_real[p] = X_test[f'{p}_lag1'].values + Y_delta_pred_df[f'{p}_delta'].values

    # ---------------- 4. 多维指标输出 (包含 MAE 和 SMAPE) ----------------
    print("\n" + "=" * 100)
    print("🏆 升级版模型竞技场 (基准 Baseline VS 全气象 RF)")
    print("=" * 100)
    print(
        f"{'污染物':<6} | {'基准 R2':<7} | {'模型 R2':<7} | {'基准 RMSE':<9} | {'模型 RMSE':<9} | {'MAE':<6} | {'sMAPE(%)':<8} | {'误差降低'}")
    print("-" * 100)

    results_data = []
    for p in pollutants:
        y_t = Y_test_real[p].values
        y_p = Y_pred_real[p].values
        y_b = X_test[f'{p}_lag1'].values

        # 统一量纲计算 (CO 除以 1000 回归 mg/m3)
        factor = 1000 if p == 'CO' else 1
        y_t_mg, y_p_mg, y_b_mg = y_t / factor, y_p / factor, y_b / factor

        r2_m = r2_score(y_t_mg, y_p_mg)
        r2_b = r2_score(y_t_mg, y_b_mg)
        rmse_m = np.sqrt(mean_squared_error(y_t_mg, y_p_mg))
        rmse_b = np.sqrt(mean_squared_error(y_t_mg, y_b_mg))
        mae_m = mean_absolute_error(y_t_mg, y_p_mg)
        smape_m = smape(y_t_mg, y_p_mg)
        imp = (rmse_b - rmse_m) / rmse_b * 100

        print(
            f"[{p:<5}] | {r2_b:<8.3f} | {r2_m:<8.3f} | {rmse_b:<10.2f} | {rmse_m:<10.2f} | {mae_m:<6.2f} | {smape_m:<8.2f} | {imp:.1f}%")

        results_data.append({
            '污染物': p,
            '模型 R2': r2_m,
            '基准 R2': r2_b,
            '模型 RMSE': rmse_m,
            '基准 RMSE': rmse_b,
            '模型 MAE': mae_m,
            '模型 sMAPE(%)': smape_m,
            '误差降低率(%)': imp
        })
    print("=" * 100)

    # 🔧 修复 3：一键导出论文排版表格
    pd.DataFrame(results_data).to_csv('RF_Model_Results_V2.csv', index=False, encoding='utf-8-sig')
    print("✅ 指标大表已自动保存至 'RF_Model_Results_V2.csv'！")

    # ---------------- 5. 协同验证图 ----------------
    print("\n" + "=" * 55)
    print("📊 多污染物协同网络拓扑误差 (Frobenius 范数)")
    print("=" * 55)

    # 图 1：绝对浓度协同
    corr_true = Y_test_real.corr()
    corr_pred_rf = Y_pred_real.corr()

    # 【核心3】提取阶段三（RF-ΔY）的 F-dist
    frob_norm_rf = np.linalg.norm(corr_true.values - corr_pred_rf.values)

    print(f"🎯 [阶段三] 本文模型 (RF-ΔY) F-dist : {frob_norm_rf:.4f}")
    print("=" * 55)


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(corr_true, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0], fmt='.2f')
    axes[0].set_title("真实浓度相关性矩阵")
    sns.heatmap(corr_pred_rf, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1], fmt='.2f')
    axes[1].set_title(f"模型预测相关性矩阵 (F-dist: {frob_norm_rf:.4f})")
    plt.tight_layout()
    plt.savefig('Q2_Synergy_Correlation_V2.png', dpi=300)

    # 图 2：ΔY 变化量协同
    delta_corr_true = Y_delta_test.corr()
    delta_corr_pred = Y_delta_pred_df.corr()
    frob_delta = np.linalg.norm(delta_corr_true.values - delta_corr_pred.values)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(delta_corr_true, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes2[0], fmt='.2f',
                xticklabels=pollutants, yticklabels=pollutants)
    axes2[0].set_title("真实浓度变化量(ΔY)协同矩阵")
    sns.heatmap(delta_corr_pred, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes2[1], fmt='.2f',
                xticklabels=pollutants, yticklabels=pollutants)
    axes2[1].set_title(f"模型预测变化量(ΔY)协同矩阵 (F-dist: {frob_delta:.4f})")
    plt.tight_layout()
    plt.savefig('Q2_Synergy_Delta_Correlation_V2.png', dpi=300)

    print("✅ 两张热力图已生成完毕！完美收工！")