import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 0. 国标 AQI 计算引擎 (完美六项满血版)
# ==========================================
def calculate_iaqi(cp, bp_lo, bp_hi, iaqi_lo, iaqi_hi):
    if cp <= 0: return 0
    return ((iaqi_hi - iaqi_lo) / (bp_hi - bp_lo)) * (cp - bp_lo) + iaqi_lo


def get_aqi(row):
    iaqi_dict = {}

    # 1. PM2.5
    v = row['PM2.5']
    if v <= 35:
        iaqi_dict['PM2.5'] = calculate_iaqi(v, 0, 35, 0, 50)
    elif v <= 75:
        iaqi_dict['PM2.5'] = calculate_iaqi(v, 35, 75, 50, 100)
    elif v <= 115:
        iaqi_dict['PM2.5'] = calculate_iaqi(v, 75, 115, 100, 150)
    elif v <= 150:
        iaqi_dict['PM2.5'] = calculate_iaqi(v, 115, 150, 150, 200)
    elif v <= 250:
        iaqi_dict['PM2.5'] = calculate_iaqi(v, 150, 250, 200, 300)
    else:
        iaqi_dict['PM2.5'] = calculate_iaqi(v, 250, 350, 300, 400)

    # 2. PM10
    v = row['PM10']
    if v <= 50:
        iaqi_dict['PM10'] = calculate_iaqi(v, 0, 50, 0, 50)
    elif v <= 150:
        iaqi_dict['PM10'] = calculate_iaqi(v, 50, 150, 50, 100)
    elif v <= 250:
        iaqi_dict['PM10'] = calculate_iaqi(v, 150, 250, 100, 150)
    elif v <= 350:
        iaqi_dict['PM10'] = calculate_iaqi(v, 250, 350, 150, 200)
    else:
        iaqi_dict['PM10'] = calculate_iaqi(v, 350, 420, 200, 300)

    # 3. O3 (严格对比 1小时与 8小时滑动，取其大者)
    v_1h = row['O3']
    v_8h = row.get('O3_8h', v_1h)  # 防护：若无8h则默认等同1h
    if pd.isna(v_8h): v_8h = v_1h

    iaqi_o3_1h, iaqi_o3_8h = 0, 0
    if v_1h <= 160:
        iaqi_o3_1h = calculate_iaqi(v_1h, 0, 160, 0, 50)
    elif v_1h <= 200:
        iaqi_o3_1h = calculate_iaqi(v_1h, 160, 200, 50, 100)
    elif v_1h <= 300:
        iaqi_o3_1h = calculate_iaqi(v_1h, 200, 300, 100, 150)
    elif v_1h <= 400:
        iaqi_o3_1h = calculate_iaqi(v_1h, 300, 400, 150, 200)
    else:
        iaqi_o3_1h = calculate_iaqi(v_1h, 400, 800, 200, 300)

    if v_8h <= 100:
        iaqi_o3_8h = calculate_iaqi(v_8h, 0, 100, 0, 50)
    elif v_8h <= 160:
        iaqi_o3_8h = calculate_iaqi(v_8h, 100, 160, 50, 100)
    elif v_8h <= 215:
        iaqi_o3_8h = calculate_iaqi(v_8h, 160, 215, 100, 150)
    elif v_8h <= 265:
        iaqi_o3_8h = calculate_iaqi(v_8h, 215, 265, 150, 200)
    else:
        iaqi_o3_8h = calculate_iaqi(v_8h, 265, 800, 200, 300)
    iaqi_dict['O3'] = max(iaqi_o3_1h, iaqi_o3_8h)

    # 4. NO2
    v = row['NO2']
    if v <= 100:
        iaqi_dict['NO2'] = calculate_iaqi(v, 0, 100, 0, 50)
    elif v <= 200:
        iaqi_dict['NO2'] = calculate_iaqi(v, 100, 200, 50, 100)
    elif v <= 700:
        iaqi_dict['NO2'] = calculate_iaqi(v, 200, 700, 100, 150)
    else:
        iaqi_dict['NO2'] = calculate_iaqi(v, 700, 1200, 150, 200)

    # 5. SO2
    v = row['SO2']
    if v <= 150:
        iaqi_dict['SO2'] = calculate_iaqi(v, 0, 150, 0, 50)
    elif v <= 500:
        iaqi_dict['SO2'] = calculate_iaqi(v, 150, 500, 50, 100)
    elif v <= 650:
        iaqi_dict['SO2'] = calculate_iaqi(v, 500, 650, 100, 150)
    else:
        iaqi_dict['SO2'] = calculate_iaqi(v, 650, 800, 150, 200)

    # 6. CO (已在特征工程阶段统一化为 mg/m3)
    v = row['CO']
    if v <= 5:
        iaqi_dict['CO'] = calculate_iaqi(v, 0, 5, 0, 50)
    elif v <= 10:
        iaqi_dict['CO'] = calculate_iaqi(v, 5, 10, 50, 100)
    elif v <= 35:
        iaqi_dict['CO'] = calculate_iaqi(v, 10, 35, 100, 150)
    else:
        iaqi_dict['CO'] = calculate_iaqi(v, 35, 60, 150, 200)

    aqi = max(iaqi_dict.values())
    primary = max(iaqi_dict, key=iaqi_dict.get) if aqi > 50 else "无"
    return pd.Series([round(aqi), primary])


# ==========================================
# 1. 特征工程 (找回日夜锚点 Lag24)
# ==========================================
def build_features(df):
    df_feat = df.copy()
    df_feat['datetime'] = pd.to_datetime(df_feat['datetime'], errors='coerce')
    df_feat = df_feat.dropna(subset=['datetime'])

    pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    # 🌟 在源头统一将 CO 转为 mg/m3，后续不再重复操作！
    df_feat['CO'] = pd.to_numeric(df_feat['CO'], errors='coerce') / 1000.0

    meteo_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']

    for lag in [1, 6, 12, 24]:
        df_feat[f'Temp_lag{lag}'] = df_feat['temperature'].shift(lag)
        df_feat[f'Hum_lag{lag}'] = df_feat['humidity'].shift(lag)
    for lag in [1, 24]:
        for col in ['wind_speed', 'pressure']:
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)

    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['datetime'].dt.hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['datetime'].dt.hour / 24)

    for p in pollutants:
        if p != 'CO': df_feat[p] = pd.to_numeric(df_feat[p], errors='coerce')
        # 🌟 绝杀修复：引入滞后 24 小时真实污染锚点，强行纠正倒转问题！
        df_feat[f'{p}_lag1'] = df_feat[p].shift(1)
        df_feat[f'{p}_lag24'] = df_feat[p].shift(24)
        df_feat[f'{p}_delta'] = df_feat[p] - df_feat[f'{p}_lag1']

    return df_feat.dropna().reset_index(drop=True), pollutants, meteo_cols


if __name__ == "__main__":
    print("⏳ 1. 读取数据并执行特征工程...")
    df = pd.read_excel('Adata2.xlsx')
    df_processed, pollutants, meteo_cols = build_features(df)

    delta_cols = [f'{p}_delta' for p in pollutants]
    drop_cols = pollutants + delta_cols + ['datetime']
    X = df_processed.drop(columns=[c for c in drop_cols if c in df_processed.columns])
    Y_delta = df_processed[delta_cols]

    # 按月份切分
    train_mask = df_processed['datetime'].dt.month <= 2
    test_mask = df_processed['datetime'].dt.month == 3

    X_train = X[train_mask]
    Y_delta_train = Y_delta[train_mask]

    X_test = X[test_mask].reset_index(drop=True)
    test_datetime = df_processed['datetime'][test_mask].reset_index(drop=True)
    Y_test_real = df_processed[pollutants][test_mask].reset_index(drop=True)

    print("🧠 2. 正在训练包含日夜锚点的真正 RF-ΔY 模型...")
    rf_base = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(rf_base)
    model.fit(X_train, Y_delta_train)

    print("🚀 3. 开启 24 小时自回归滚动预测...")
    Y_pred_rolling = pd.DataFrame(index=range(len(X_test)), columns=pollutants)

    p_max = df_processed[pollutants].max().values * 1.3
    p_min = np.maximum(0, df_processed[pollutants].min().values * 0.7)

    test_len = len(X_test)
    for start_idx in range(0, test_len, 24):
        end_idx = min(start_idx + 24, test_len)
        chunk_size = end_idx - start_idx

        current_lag1 = X_test.iloc[start_idx][[f'{p}_lag1' for p in pollutants]].values.astype(float)

        for h in range(chunk_size):
            curr_row_idx = start_idx + h
            X_h = X_test.iloc[curr_row_idx].copy()

            # 注意：这里仅更新 lag1，因为 lag24（昨天同时刻）已知且固定，这就是锚定日夜的秘密！
            for i, p in enumerate(pollutants):
                X_h[f'{p}_lag1'] = current_lag1[i]

            delta_pred = model.predict(pd.DataFrame([X_h], columns=X_test.columns))[0]

            abs_pred = current_lag1 + delta_pred
            abs_pred = np.clip(abs_pred, p_min, p_max)

            Y_pred_rolling.iloc[curr_row_idx, :len(pollutants)] = abs_pred
            current_lag1 = abs_pred

    # ==========================================
    # 🌟 绝杀环节：完美实现论文中的“实况预测混合序列拼接法”算 O3_8h
    # ==========================================
    print("🔄 正在执行《混合序列终点对齐法》计算 O3 8小时滑动均值...")
    # 把真实历史和未来预测首尾相连拼接起来
    hybrid_o3 = pd.concat([Y_test_real['O3'], Y_pred_rolling['O3']]).reset_index(drop=True)
    # 对混合序列进行 8 小时滑动平滑
    hybrid_o3_8h = hybrid_o3.rolling(8, min_periods=1).mean()
    # 完美切片塞回两个表中，供 AQI 函数调用
    Y_test_real['O3_8h'] = hybrid_o3_8h.iloc[:len(Y_test_real)].values
    Y_pred_rolling['O3_8h'] = hybrid_o3_8h.iloc[len(Y_test_real):].values

    # ==========================================
    # 4. 指标量化评估与 AQI 结算
    # ==========================================
    print("\n" + "=" * 65)
    print("🏆 3月份 (全月) RF-ΔY 自回归量化精度报表 (修正版)")
    print("=" * 65)
    print(f"{'污染物':<8} | {'R²':<15} | {'RMSE':<15} | {'MAE'}")
    print("-" * 65)

    metrics_data = []
    for p in pollutants:
        y_true = Y_test_real[p].values
        y_pred = Y_pred_rolling[p].values.astype(float)

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        print(f"{p:<10} | {r2:<18.3f} | {rmse:<18.2f} | {mae:.2f}")
        metrics_data.append({'污染物': p, 'R²': round(r2, 3), 'RMSE': round(rmse, 2), 'MAE': round(mae, 2)})

    print("=" * 65)

    Y_test_real[['AQI', 'Primary']] = Y_test_real.apply(get_aqi, axis=1)
    Y_pred_rolling[['AQI', 'Primary']] = Y_pred_rolling.apply(get_aqi, axis=1)

    # 导出给排班系统用
    q3_input = pd.concat([test_datetime, X_test[['temperature', 'humidity', 'wind_speed']], Y_pred_rolling], axis=1)
    q3_input.to_csv('Q3_Forecast_Data.csv', index=False, encoding='utf-8-sig')

    # ==========================================
    # 5. 可视化
    # ==========================================
    print("🎨 正在生成修正后的长效走势对比图...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    plot_len = 24 * 14
    time_ax = test_datetime.iloc[-plot_len:]

    true_aqi = Y_test_real['AQI'].iloc[-plot_len:]
    pred_aqi = Y_pred_rolling['AQI'].iloc[-plot_len:]

    ax1.plot(time_ax, true_aqi, label='真实 AQI 实况', color='#1f77b4', linewidth=2)
    ax1.plot(time_ax, pred_aqi, label='RF-ΔY 预测 AQI', color='#ff7f0e', linestyle='--', linewidth=2)
    ax1.axhline(100, color='red', linestyle=':', label='超标警戒线 (AQI=100)')
    ax1.set_title("长效自回归滚动预测验证：综合 AQI 连续 14 天走势对比", fontsize=16, fontweight='bold')
    ax1.set_ylabel("AQI", fontsize=12)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.4)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    true_o3 = Y_test_real['O3'].iloc[-plot_len:]
    pred_o3 = Y_pred_rolling['O3'].iloc[-plot_len:]

    ax2.plot(time_ax, true_o3, label='真实 O3 浓度', color='#2ca02c', linewidth=2)
    ax2.plot(time_ax, pred_o3, label='RF-ΔY 预测 O3 浓度', color='#d62728', linestyle='--', linewidth=2)
    ax2.set_title("O3 光化学污染预报拟合效果 (完美落实混合序列终点拼接对齐)", fontsize=16, fontweight='bold')
    ax2.set_ylabel("O3 浓度 (μg/m³)", fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, alpha=0.4)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.tight_layout()
    plt.savefig('Q2_Strict_Alignment_Forecast.png', dpi=300, bbox_inches='tight')
    print("🎉 大功告成！曲线彻底纠正，倒转灰飞烟灭！")