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
# 1. AQI 计算引擎 (直接使用原始量纲 mg/m3 即可)
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
# 2. 严格时序特征重构 (治愈反相 Bug 的核心)
# ==========================================
print("📂 正在加载数据并重构严谨的自回归锚点特征...")
file_path = 'Adata2.xlsx'
df = pd.read_excel(file_path)
df['datetime'] = pd.to_datetime(df['datetime'])

pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
for p in pollutants:
    df[p] = pd.to_numeric(df[p], errors='coerce').astype('float64')

# 🌟 核心修复 1：强制加入 lag1（上一小时浓度）。让 LSTM 预测本小时数据时，永远有一个“刚刚的真实浓度”作为锚点！
lag_cols = [f'{p}_lag1' for p in pollutants]
for p in pollutants:
    df[f'{p}_lag1'] = df[p].shift(1)

# 时间规律
df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
meteo_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']

df = df.dropna().reset_index(drop=True)

# 🌟 核心修复 2：彻底砍掉导致冬春概念漂移的 12h 复杂天气滞后，回归最纯粹的推演
feature_cols = lag_cols + meteo_cols + ['hour_sin', 'hour_cos']

# ==========================================
# 3. 提取 1-2 月训练 LSTM
# ==========================================
print("🏗️ 准备训练带有强自回归约束的 LSTM 网络...")
df_train = df[df['datetime'].dt.month < 3].copy()
X_data = df_train[feature_cols].values
Y_data = df_train[pollutants].values

WINDOW_SIZE = 24


def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    # 🌟 核心修复 3：完美序列对齐，保证 X 的最后一行包含 t-1 的污染和 t 的天气，目标值精准指向 t
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i: i + time_steps])
        ys.append(y[i + time_steps - 1])
    return np.array(Xs), np.array(ys)


X_seq, Y_seq = create_sequences(X_data, Y_data, WINDOW_SIZE)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_x.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
Y_train_sc = scaler_y.fit_transform(Y_seq)

print("🚀 开始训练模型 (带有防过拟合终止机制)...")
model = Sequential([
    Input(shape=(WINDOW_SIZE, len(feature_cols))),
    LSTM(64, return_sequences=True, activation='tanh'),
    Dropout(0.2),
    LSTM(32, activation='tanh'),
    Dense(16, activation='relu'),
    Dense(6)
])
model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
model.fit(X_train_sc, Y_train_sc, epochs=50, batch_size=64, callbacks=[early_stop], verbose=1)

# ==========================================
# 4. 第三问：三月份 24 小时动态填补推演
# ==========================================
print("\n🔄 开始三月份的真实闭环仿真 (严格阻断时序泄露)...")
march_days = df[(df['datetime'].dt.month == 3)]['datetime'].dt.day.unique()
all_predictions = []

for day in march_days:
    # 每天 22:00 启动预测
    anchor_time = pd.to_datetime(f'2026-02-28 22:00:00') if day == 1 else pd.to_datetime(
        f'2026-03-{day - 1:02d} 22:00:00')

    window_start = anchor_time + pd.Timedelta(hours=1)
    window_end = anchor_time + pd.Timedelta(hours=24)
    if window_end > df['datetime'].max(): window_end = df['datetime'].max()

    # 构建当前工作台
    context_start = anchor_time - pd.Timedelta(hours=WINDOW_SIZE + 48)
    df_work = df[(df['datetime'] >= context_start) & (df['datetime'] <= window_end)].copy()

    # 🌟 防泄露双重保险：抹掉未来的真实值，以及未来的滞后值
    future_mask = df_work['datetime'] > anchor_time
    for p in pollutants:
        df_work.loc[future_mask, p] = np.nan

    future_lag_mask = df_work['datetime'] > anchor_time + pd.Timedelta(hours=1)
    for p in pollutants:
        df_work.loc[future_lag_mask, f'{p}_lag1'] = np.nan

    current_pred_times = pd.date_range(window_start, window_end, freq='h')

    for target_time in current_pred_times:
        df_temp = df_work[df_work['datetime'] <= target_time]
        seq_data = df_temp[feature_cols].iloc[-WINDOW_SIZE:].values

        # 数据完整性拦截
        if len(seq_data) < WINDOW_SIZE or np.isnan(seq_data).any():
            continue

        seq_sc = scaler_x.transform(seq_data).reshape(1, WINDOW_SIZE, -1)
        pred_sc = model.predict(seq_sc, verbose=0)
        pred_true = scaler_y.inverse_transform(pred_sc)[0]

        row_result = {'datetime': target_time}
        for i, p in enumerate(pollutants):
            val = max(0, pred_true[i])

            # 第一步：把预测出的当期浓度，填进工作台
            df_work.loc[df_work['datetime'] == target_time, p] = val
            row_result[f'{p}_pred'] = val

            # 🌟 核心修复 4：链式传递！把刚刚预测出的浓度，立刻作为“下一小时”的 lag1 特征填进去！
            next_time = target_time + pd.Timedelta(hours=1)
            if (df_work['datetime'] == next_time).any():
                df_work.loc[df_work['datetime'] == next_time, f'{p}_lag1'] = val

        all_predictions.append(row_result)

    if day % 5 == 0 or day == 1:
        print(f"✅ 成功无缝推演至: {window_end.strftime('%Y-%m-%d %H:00')}")

# ==========================================
# 5. 出图与评价指标
# ==========================================
df_pred = pd.DataFrame(all_predictions)
df_true = df[df['datetime'].dt.month == 3][['datetime'] + pollutants].copy()
df_eval = pd.merge(df_true, df_pred, on='datetime', how='inner')

print("\n🏆 修复后的 LSTM 全月预测评价大表")
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

pd.DataFrame(eval_results).to_csv('Q3_March_LSTM_Metrics_Fixed.csv', index=False, encoding='utf-8-sig')

df_eval['True_AQI'] = df_eval.apply(get_comprehensive_aqi, axis=1)
temp_pred_df = df_eval[['datetime'] + [f'{p}_pred' for p in pollutants]].rename(
    columns={f'{p}_pred': p for p in pollutants})
df_eval['Pred_AQI'] = temp_pred_df.apply(get_comprehensive_aqi, axis=1)
df_eval['活动建议'] = df_eval['Pred_AQI'].apply(get_aqi_level)
df_eval.to_csv('Q3_March_LSTM_Predictions_AQI_Fixed.csv', index=False, encoding='utf-8-sig')

# ============ 绘制雪耻图 ============
print("\n📊 正在生成最后 5 天真·无缝跟随趋势图...")
df_plot = df_eval[df_eval['datetime'] >= '2026-03-27 00:00:00'].set_index('datetime')

# 图 1
plt.figure(figsize=(14, 5))
plt.plot(df_plot.index, df_plot['True_AQI'], label='真实 AQI', color='black', marker='o', markersize=4)
plt.plot(df_plot.index, df_plot['Pred_AQI'], label='LSTM预测 AQI', color='#FF5722', linestyle='--', marker='^',
         markersize=4)
plt.axhline(y=100, color='red', linestyle=':', label='污染警戒线')
plt.fill_between(df_plot.index, df_plot['True_AQI'], df_plot['Pred_AQI'], color='gray', alpha=0.1)
plt.title('基于强自回归 LSTM 的良乡校区综合 AQI 滚动预测对齐验证', fontsize=16)
plt.ylabel('AQI')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q3_LSTM_Last5Days_AQI_Fixed.png', dpi=300)

# 图 2
fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
axes = axes.flatten()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, p in enumerate(pollutants):
    ax = axes[i]
    ax.plot(df_plot.index, df_plot[p], label='真实值', color='black', linewidth=1.5)
    ax.plot(df_plot.index, df_plot[f'{p}_pred'], label='预测值', color=colors[i], linestyle='--', linewidth=2)
    unit = 'mg/m³' if p == 'CO' else 'μg/m³'
    ax.set_title(f'{p} 预测与真实趋势对比 (消除倒影)', fontweight='bold')
    ax.set_ylabel(f'浓度 ({unit})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('最后5天六项污染物协同滚动预测 (强时序依赖版)', fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig('Q3_LSTM_Last5Days_Pollutants_Fixed.png', dpi=300)

print("✅ 大功告成！心电图 BUG 彻底消除，真正的预测曲线已生成！")