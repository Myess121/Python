import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据加载与预处理 (已融合风压数据)
# ==========================================
print("📂 正在加载多源气象数据...")
file_path = 'Adata2.xlsx'  # 确保你的文件叫这个名字
df = pd.read_excel(file_path)
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime']).reset_index(drop=True)

pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
df['CO'] = pd.to_numeric(df['CO'], errors='coerce') * 1000  # 统一为 ug/m3

# ==========================================
# 2. 特征工程
# ==========================================
for p in pollutants:
    df[f'{p}_lag1'] = df[p].shift(1)
    df[f'{p}_delta'] = df[p] - df[f'{p}_lag1']

df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)

# 🌟 多源气象特征库
meteo_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']
for lag in [1, 6, 12]:
    for col in meteo_cols:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

df = df.dropna().reset_index(drop=True)

delta_cols = [f'{p}_delta' for p in pollutants]
lag1_cols = [f'{p}_lag1' for p in pollutants]

feature_cols = meteo_cols + ['hour_sin', 'hour_cos'] + \
               [f'{c}_lag{l}' for l in [1, 6, 12] for c in meteo_cols] + lag1_cols

# 转为 Numpy 数组，避免切片 Bug
X_data = df[feature_cols].values
Y_data = df[delta_cols].values


# ==========================================
# 3. 严格防泄露划分 & 序列重构 (千问神补丁)
# ==========================================
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


WINDOW_SIZE = 24
X_seq, Y_seq = create_sequences(X_data, Y_data, WINDOW_SIZE)

split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
Y_train, Y_test = Y_seq[:split], Y_seq[split:]

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_x.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test_sc = scaler_x.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
Y_train_sc = scaler_y.fit_transform(Y_train)
Y_test_sc = scaler_y.transform(Y_test)

# ==========================================
# 4. 模型构建与训练
# ==========================================
print("🏗️ 训练多源气象驱动的 LSTM 模型...")
model = Sequential([
    Input(shape=(WINDOW_SIZE, X_train_sc.shape[2])),
    LSTM(100, return_sequences=True, activation='tanh'),
    Dropout(0.2),
    LSTM(50, activation='tanh'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(6)
])
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
model.fit(X_train_sc, Y_train_sc, epochs=100, batch_size=64, validation_split=0.15, callbacks=[early_stop], verbose=1)

# ==========================================
# 5. 还原与多维指标评估 (保存 CSV 表格)
# ==========================================
Y_pred_sc = model.predict(X_test_sc)
Y_pred_delta = scaler_y.inverse_transform(Y_pred_sc)
Y_true_delta = scaler_y.inverse_transform(Y_test_sc)

n_test = len(X_test)
Y_test_real = df[pollutants].iloc[-n_test:].values
Y_test_lag1 = df[lag1_cols].iloc[-n_test:].values
Y_pred_real = Y_test_lag1 + Y_pred_delta

# CO 单位还原 (ug -> mg)
idx_co = pollutants.index('CO')
Y_test_real[:, idx_co] /= 1000
Y_pred_real[:, idx_co] /= 1000
Y_test_lag1[:, idx_co] /= 1000


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1e-8
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100


print("\n🏆 模型竞技场 (Baseline VS 多源 LSTM)")
print("=" * 95)
print(
    f"{'污染物':<8} | {'基准 R2':<8} | {'模型 R2':<8} | {'基准 RMSE':<10} | {'模型 RMSE':<10} | {'误差降低':<10} | {'sMAPE(%)':<10}")
print("-" * 95)

results_data = []
for i, p in enumerate(pollutants):
    y_t, y_p, y_b = Y_test_real[:, i], Y_pred_real[:, i], Y_test_lag1[:, i]

    r2_m, r2_b = r2_score(y_t, y_p), r2_score(y_t, y_b)
    rmse_m, rmse_b = np.sqrt(mean_squared_error(y_t, y_p)), np.sqrt(mean_squared_error(y_t, y_b))
    smape_val = smape(y_t, y_p)
    imp = (rmse_b - rmse_m) / rmse_b * 100

    print(
        f"{p:<8} | {r2_b:<8.3f} | {r2_m:<8.3f} | {rmse_b:<10.2f} | {rmse_m:<10.2f} | {imp:>7.1f}%     | {smape_val:<10.2f}")

    results_data.append({
        '污染物': p, '本文 R2': r2_m, '基准 R2': r2_b,
        '本文 RMSE': rmse_m, '基准 RMSE': rmse_b,
        '误差降低率(%)': imp, 'sMAPE(%)': smape_val
    })
print("=" * 95)

# ✅ 这里会保存出表格
pd.DataFrame(results_data).to_csv('LSTM_Comparison_Results_Final.csv', index=False, encoding='utf-8-sig')

# ==========================================
# 6. 协同矩阵验证 (保存两张热力图)
# ==========================================
# 图1: 绝对浓度协同
fig1, ax1 = plt.subplots(1, 2, figsize=(15, 6))
corr_true_abs = pd.DataFrame(Y_test_real, columns=pollutants).corr()
corr_pred_abs = pd.DataFrame(Y_pred_real, columns=pollutants).corr()
f_dist_abs = np.linalg.norm(corr_true_abs.values - corr_pred_abs.values)

sns.heatmap(corr_true_abs, annot=True, cmap='coolwarm', ax=ax1[0], fmt='.2f')
ax1[0].set_title("真实浓度相关性矩阵")
sns.heatmap(corr_pred_abs, annot=True, cmap='coolwarm', ax=ax1[1], fmt='.2f')
ax1[1].set_title(f"LSTM 预测浓度协同矩阵\n(Frobenius 距离: {f_dist_abs:.4f})")
plt.tight_layout()
plt.savefig('LSTM_Absolute_Synergy_Final.png', dpi=300)  # ✅ 这里保存第一张图

# 图2: 动态变化(ΔY)协同
fig2, ax2 = plt.subplots(1, 2, figsize=(15, 6))
corr_true_delta = pd.DataFrame(Y_true_delta, columns=pollutants).corr()
corr_pred_delta = pd.DataFrame(Y_pred_delta, columns=pollutants).corr()
f_dist_delta = np.linalg.norm(corr_true_delta.values - corr_pred_delta.values)

sns.heatmap(corr_true_delta, annot=True, cmap='coolwarm', ax=ax2[0], fmt='.2f')
ax2[0].set_title("真实变化量 (ΔY) 协同矩阵")
sns.heatmap(corr_pred_delta, annot=True, cmap='coolwarm', ax=ax2[1], fmt='.2f')
ax2[1].set_title(f"LSTM 预测变化量协同矩阵\n(Frobenius 距离: {f_dist_delta:.4f})")
plt.tight_layout()
plt.savefig('LSTM_Delta_Synergy_Final.png', dpi=300)  # ✅ 这里保存第二张图

print("\n✅ 运行完毕！CSV 和 两张热力图 已保存在当前文件夹下。")