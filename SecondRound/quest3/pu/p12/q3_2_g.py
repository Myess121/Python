import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 0. 国标 AQI 计算引擎 (严格补齐六项)
# ==========================================
def get_aqi_full(row):
    def calc_iaqi(cp, bp_lo, bp_hi, iaqi_lo, iaqi_hi):
        return ((iaqi_hi - iaqi_lo) / (bp_hi - bp_lo)) * (cp - bp_lo) + iaqi_lo

    limits = {
        'PM2.5': [0, 35, 75, 115, 150, 250, 350, 500],
        'PM10': [0, 50, 150, 250, 350, 420, 500, 600],
        'NO2': [0, 100, 200, 700, 1200, 2340, 3090, 3840],
        'SO2': [0, 150, 500, 650, 800, 1600, 2100, 2620],
        'CO': [0, 5, 10, 35, 60, 90, 120, 150]
    }
    levels = [0, 50, 100, 150, 200, 300, 400, 500]
    iaqi_dict = {}

    for p in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']:
        cp, bp = row[p], limits[p]
        idx = np.searchsorted(bp, cp)
        if idx == 0:
            iaqi_dict[p] = 0
        elif idx >= len(bp):
            iaqi_dict[p] = 500
        else:
            iaqi_dict[p] = calc_iaqi(cp, bp[idx - 1], bp[idx], levels[idx - 1], levels[idx])

    # O3 1h 与 8h 滑动均值对比取大值
    o1, o8 = row['O3'], row.get('O3_8h', row['O3'])
    bp1, bp8 = [0, 160, 200, 300, 400, 800, 1000, 1200], [0, 100, 160, 215, 265, 800, 1000, 1200]
    i1 = calc_iaqi(o1, bp1[np.searchsorted(bp1, o1) - 1], bp1[np.searchsorted(bp1, o1)],
                   levels[np.searchsorted(bp1, o1) - 1], levels[np.searchsorted(bp1, o1)]) if o1 <= 800 else 300
    i8 = calc_iaqi(o8, bp8[np.searchsorted(bp8, o8) - 1], bp8[np.searchsorted(bp8, o8)],
                   levels[np.searchsorted(bp8, o8) - 1], levels[np.searchsorted(bp8, o8)]) if o8 <= 800 else 300
    iaqi_dict['O3'] = max(i1, i8)

    aqi = max(iaqi_dict.values())
    return pd.Series([round(aqi), max(iaqi_dict, key=iaqi_dict.get) if aqi > 50 else "无"])


# ==========================================
# 1. 特征工程 (参考 rf2_n_3 逻辑进行加固)
# ==========================================
def build_robust_features(df):
    df_feat = df.copy()
    df_feat['datetime'] = pd.to_datetime(df_feat['datetime'])
    # 🌟 修复 1：强制排序并处理重复时间戳[cite: 1]
    df_feat = df_feat.sort_values('datetime').groupby('datetime').mean().reset_index()
    # 🌟 修复 2：重采样确保时间轴无间断，锁定 shift(24) 的物理含义[cite: 1]
    df_feat = df_feat.set_index('datetime').resample('1h').interpolate(method='linear').reset_index()

    pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    if df_feat['CO'].max() > 100: df_feat['CO'] /= 1000.0  # 统一为 mg/m3[cite: 1]

    # 构建气象与污染滞后特征 (对齐参考代码)[cite: 2]
    for lag in [1, 12, 24]:
        for col in pollutants + ['temperature', 'humidity', 'wind_speed', 'pressure']:
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)

    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['datetime'].dt.hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['datetime'].dt.hour / 24)

    # 差分目标计算[cite: 2]
    for p in pollutants:
        df_feat[f'{p}_delta'] = df_feat[p] - df_feat[f'{p}_lag1']

    return df_feat.dropna().reset_index(drop=True), pollutants


if __name__ == "__main__":
    print("⏳ 1. 正在按参考逻辑构建强固特征空间...")
    df_raw = pd.read_excel('Adata2.xlsx')
    df_proc, pollutants = build_robust_features(df_raw)

    delta_cols = [f'{p}_delta' for p in pollutants]
    # 🌟 修复 3：明确特征列表，确保训练与预测列序 100% 一致[cite: 1, 2]
    feature_list = [c for c in df_proc.columns if c not in pollutants + delta_cols + ['datetime']]

    X, Y_delta = df_proc[feature_list], df_proc[delta_cols]

    # 按自然月切分：1-2月训练，3月全月验证[cite: 1]
    train_idx = df_proc['datetime'].dt.month <= 2
    test_idx = df_proc['datetime'].dt.month == 3

    X_train, Y_train = X[train_idx], Y_delta[train_idx]
    X_test, Y_test_real = X[test_idx].reset_index(drop=True), df_proc[pollutants][test_idx].reset_index(drop=True)
    test_time = df_proc['datetime'][test_idx].reset_index(drop=True)

    print(f"🧠 2. 训练 RF-ΔY 多输出模型 (特征维度: {len(feature_list)})...")
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    model.fit(X_train, Y_train)

    print("🚀 3. 执行 24 小时自回归滚动预测 (名称对齐模式)...")
    Y_pred = pd.DataFrame(index=range(len(X_test)), columns=pollutants)

    for start in range(0, len(X_test), 24):
        end = min(start + 24, len(X_test))
        # 获取起点真实滞后值
        curr_lag1 = X_test.loc[start, [f'{p}_lag1' for p in pollutants]].values.astype(float)

        for h in range(start, end):
            # 🌟 核心：构造带有列名的 DataFrame 进行预测，防止位置错位[cite: 1]
            row_df = pd.DataFrame([X_test.iloc[h]], columns=feature_list)
            for i, p in enumerate(pollutants): row_df[f'{p}_lag1'] = curr_lag1[i]

            delta = model.predict(row_df)[0]
            curr_lag1 = np.maximum(0, curr_lag1 + delta)  # 物理约束：浓度非负[cite: 1]
            Y_pred.iloc[h] = curr_lag1

    print("📊 4. 正在进行全月精度量化评估...")
    # O3 8小时滑动处理 (混合序列法)[cite: 1]
    o3_all = pd.concat([Y_test_real['O3'], Y_pred['O3']]).rolling(8, min_periods=1).mean()
    Y_test_real['O3_8h'], Y_pred['O3_8h'] = o3_all.iloc[:len(Y_test_real)].values, o3_all.iloc[len(Y_test_real):].values

    # 计算准确率与 R2[cite: 1]
    accuracy_list = []
    for p in pollutants:
        yt, yp = Y_test_real[p].values, Y_pred[p].values.astype(float)
        r2 = r2_score(yt, yp)
        mape = np.mean(np.abs((yt - yp) / (yt + 1e-5))) * 100
        acc = max(0, 100 - mape)
        print(f"{p:<6} | 预测准确率: {acc:>6.2f}% | R2: {r2:>6.3f}")
        accuracy_list.append({'污染物': p, '准确率': f"{acc:.2f}%", 'R2': round(r2, 3)})

    # 计算 AQI 并绘图
    Y_test_real[['AQI', 'P']], Y_pred[['AQI', 'P']] = Y_test_real.apply(get_aqi_full, axis=1), Y_pred.apply(
        get_aqi_full, axis=1)
    pd.DataFrame(accuracy_list).to_csv('Q2_Final_Accuracy.csv', index=False, encoding='utf-8-sig')

    plt.figure(figsize=(15, 6))
    plt.plot(test_time, Y_test_real['AQI'], label='真实值', color='#1f77b4', alpha=0.8)
    plt.plot(test_time, Y_pred['AQI'], label='预测值(滚动)', color='#ff7f0e', linestyle='--', alpha=0.9)
    plt.title("3月全月空气质量指数 (AQI) 滚动预测验证 - 逻辑修正版")
    plt.legend()
    plt.savefig('Final_Validation_No_Inversion.png', dpi=300)
    print("✅ 任务彻底完成！请检查图表 Final_Validation_No_Inversion.png 是否已纠正相位。")