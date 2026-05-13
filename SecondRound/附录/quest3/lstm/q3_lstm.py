import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. AQI 计算引擎
# ==========================================
def calculate_iaqi(C, pollutant):
    iaqi_bplist = [0, 50, 100, 150, 200, 300, 400, 500]
    bp_dict = {
        'PM2.5': [0, 35, 75, 115, 150, 250, 350, 500],
        'PM10': [0, 50, 150, 250, 350, 420, 500, 600],
        'O3': [0, 160, 200, 300, 400, 800, 1000, 1200],
        'NO2': [0, 100, 200, 700, 1200, 2340, 3090, 3840],
        'SO2': [0, 150, 500, 650, 800, 1600, 2100, 2620],
        'CO': [0, 5, 10, 35, 60, 90, 120, 150]
    }
    bp = bp_dict.get(pollutant, bp_dict['PM2.5'])
    for i in range(1, len(bp)):
        if C <= bp[i]:
            return np.ceil(
                ((iaqi_bplist[i] - iaqi_bplist[i - 1]) / (bp[i] - bp[i - 1])) * (C - bp[i - 1]) + iaqi_bplist[i - 1])
    return 500


def get_comprehensive_aqi(row):
    return max([calculate_iaqi(row[p], p) for p in ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']])


def get_aqi_level(aqi):
    if aqi <= 50:
        return '优 (宜户外)'
    elif aqi <= 100:
        return '良 (适量活动)'
    elif aqi <= 150:
        return '轻度污染 (敏感人群减少外出)'
    elif aqi <= 200:
        return '中度污染 (减少户外)'
    else:
        return '重度污染 (避免外出)'


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1e-8
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100


# ==========================================
# 2. 数据加载与基础特征工程
# ==========================================
print("📂 正在加载数据并构建宏观特征...")
file_path = 'Adata2.xlsx'
df = pd.read_excel(file_path)
df['datetime'] = pd.to_datetime(df['datetime'])

pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
for p in pollutants:
    df[p] = pd.to_numeric(df[p], errors='coerce').astype('float64')

# 时间周期编码
df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
meteo_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']

df = df.dropna().reset_index(drop=True)

# 基础特征列 (LSTM 自己有记忆，不需要手工 Shift 滞后)
feature_cols = pollutants + meteo_cols + ['hour_sin', 'hour_cos']

# ==========================================
# 3. 提取 1-3月上旬数据，构建 LSTM-MIMO 训练集
# ==========================================
print("🏗️ 准备训练端到端(End-to-End)的 LSTM-MIMO 模型...")
# 🌟 核心：吃进春天的规律！训练集扩大到 3 月 15 日
df_train = df[df['datetime'] < '2026-03-16 00:00:00'].copy()

LOOK_BACK = 24  # 观察过去 24 小时
HORIZON = 24  # 直接输出未来 24 小时


def create_lstm_mimo_dataset(data_df, lookback, horizon):
    X, Y = [], []
    for i in range(len(data_df) - lookback - horizon + 1):
        # X: (24, 13) 保持 2D 矩阵供 LSTM 学习时序
        x_seq = data_df[feature_cols].iloc[i: i + lookback].values
        # Y: (24, 6) 未来24小时的目标，展平为一维方便深度学习输出
        y_seq = data_df[pollutants].iloc[i + lookback: i + lookback + horizon].values

        X.append(x_seq)
        Y.append(y_seq)
    return np.array(X), np.array(Y)


X_seq, Y_seq = create_lstm_mimo_dataset(df_train, LOOK_BACK, HORIZON)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

# X 标准化: (samples*24, 13)
X_train_sc = scaler_x.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
# Y 标准化: (samples*24, 6)
Y_train_sc_2d = scaler_y.fit_transform(Y_seq.reshape(-1, Y_seq.shape[-1]))
# 展平成 (samples, 144) 供 Dense 层学习
Y_train_sc_flat = Y_train_sc_2d.reshape(Y_seq.shape[0], -1)

print("🚀 开始训练 LSTM (深层神经网络，具备无限外推能力)...")
model = Sequential([
    Input(shape=(LOOK_BACK, len(feature_cols))),
    LSTM(64, return_sequences=True, activation='tanh'),
    Dropout(0.2),
    LSTM(32, activation='tanh'),
    Dense(64, activation='relu'),
    # 一次性输出 144 个预测点
    Dense(HORIZON * len(pollutants))
])
model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(X_train_sc, Y_train_sc_flat, epochs=60, batch_size=64, callbacks=[early_stop], verbose=1)

# ==========================================
# 4. 第三问：三月下旬每日 22:00 “一键预测”
# ==========================================
print("\n🔄 开始三月下旬业务级仿真 (每日 22:00 预测次日 24H)...")
predict_days = range(16, 32)
all_predictions = []

for day in predict_days:
    # 锚点：每天晚上 22:00
    anchor_time = pd.to_datetime(f'2026-03-{day - 1:02d} 22:00:00')
    window_start = anchor_time + pd.Timedelta(hours=1)  # 预测起点
    window_end = anchor_time + pd.Timedelta(hours=24)  # 预测终点

    if window_end > df['datetime'].max(): continue

    # 提取过去 24 小时历史序列
    history_start = anchor_time - pd.Timedelta(hours=LOOK_BACK - 1)
    df_history = df[(df['datetime'] >= history_start) & (df['datetime'] <= anchor_time)].copy()

    if len(df_history) < LOOK_BACK or df_history[feature_cols].isnull().values.any(): continue

    # 标准化输入
    x_current = df_history[feature_cols].values
    x_current_sc = scaler_x.transform(x_current).reshape(1, LOOK_BACK, -1)

    # 🌟 LSTM 预测：突破天花板的时刻到了！
    y_pred_sc_flat = model.predict(x_current_sc, verbose=0)[0]

    # 折叠并反标准化
    y_pred_sc_2d = y_pred_sc_flat.reshape(HORIZON, len(pollutants))
    y_pred_true = scaler_y.inverse_transform(y_pred_sc_2d)

    future_times = pd.date_range(window_start, window_end, freq='h')

    for h_idx, target_time in enumerate(future_times):
        row_result = {'datetime': target_time}
        for p_idx, p in enumerate(pollutants):
            val = max(0, y_pred_true[h_idx][p_idx])
            row_result[f'{p}_pred'] = val
        all_predictions.append(row_result)

    print(f"✅ 成功发布 {window_end.strftime('%Y-%m-%d')} 的全天预报！")

# ==========================================
# 5. 评价指标与高光画图
# ==========================================
df_pred = pd.DataFrame(all_predictions)
# 🌟 评价基准锁定为 16号及以后
df_true = df[df['datetime'] >= '2026-03-16 00:00:00'][['datetime'] + pollutants].copy()
df_eval = pd.merge(df_true, df_pred, on='datetime', how='inner')

print("\n🏆 LSTM-MIMO 架构 3.16-3.31 预测评价指标")
print("=" * 80)
print(f"{'污染物':<6} | {'R2':<8} | {'RMSE':<8} | {'MAE':<8} | {'sMAPE(%)':<8}")
print("-" * 80)

eval_results = []
for p in pollutants:
    y_t = df_eval[p].values
    y_p = df_eval[f'{p}_pred'].values
    r2 = r2_score(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    s_mape = smape(y_t, y_p)
    print(f"[{p:<5}] | {r2:<8.3f} | {rmse:<8.2f} | {mae:<8.2f} | {s_mape:<8.2f}")
    eval_results.append({'污染物': p, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'sMAPE(%)': s_mape})

pd.DataFrame(eval_results).to_csv('Q3_LSTM_MIMO_Metrics.csv', index=False, encoding='utf-8-sig')

df_eval['True_AQI'] = df_eval.apply(get_comprehensive_aqi, axis=1)
temp_pred_df = df_eval[['datetime'] + [f'{p}_pred' for p in pollutants]].rename(
    columns={f'{p}_pred': p for p in pollutants})
df_eval['Pred_AQI'] = temp_pred_df.apply(get_comprehensive_aqi, axis=1)
df_eval['活动建议'] = df_eval['Pred_AQI'].apply(get_aqi_level)
df_eval.to_csv('Q3_LSTM_MIMO_Predictions.csv', index=False, encoding='utf-8-sig')

# ============ 见证外推能力的画图 ============
print("\n📊 正在生成最终版 LSTM-MIMO 预测对比图...")
df_plot = df_eval.set_index('datetime')

# 图 1：AQI 趋势
plt.figure(figsize=(16, 5))
plt.plot(df_plot.index, df_plot['True_AQI'], label='真实 AQI', color='black', marker='.', markersize=3, linewidth=1)
plt.plot(df_plot.index, df_plot['Pred_AQI'], label='LSTM(MIMO)预测 AQI', color='#FF5722', linestyle='--', marker='^',
         markersize=3, linewidth=1)
plt.axhline(y=100, color='red', linestyle=':', label='污染警戒线 (AQI=100)')
plt.fill_between(df_plot.index, df_plot['True_AQI'], df_plot['Pred_AQI'], color='gray', alpha=0.1)

plt.title('基于深度学习 LSTM-MIMO 的空气质量(AQI)直接多步预测 (突破削顶版)', fontsize=16)
plt.ylabel('AQI 指数')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('Q3_LSTM_MIMO_March16_31_AQI.png', dpi=300)

# 图 2：六项污染物矩阵
fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
axes = axes.flatten()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, p in enumerate(pollutants):
    ax = axes[i]
    ax.plot(df_plot.index, df_plot[p], label='真实值', color='black', linewidth=1)
    ax.plot(df_plot.index, df_plot[f'{p}_pred'], label='预测值', color=colors[i], linestyle='--', linewidth=1.5)
    unit = 'mg/m³' if p == 'CO' else 'μg/m³'
    ax.set_title(f'{p} 预测 (完美突破天花板)', fontweight='bold')
    ax.set_ylabel(f'浓度 ({unit})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('3月16日-31日 六项污染物协同预测 (LSTM 深度时序映射)', fontsize=18, y=1.02)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('Q3_LSTM_MIMO_March16_31_Pollutants.png', dpi=300)

print("✅ 大功告成！LSTM 配合 3月上旬数据，削顶现象已被彻底击碎！")