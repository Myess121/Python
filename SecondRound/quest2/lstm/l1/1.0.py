import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import warnings

# 基础设置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据加载与核心预处理
# ==========================================
print("📂 正在加载数据...")
file_path = '4-2026年校赛第二轮A题数据.xlsx'
df = pd.read_excel(file_path, skiprows=9)
df['datetime'] = df['datetime'].astype(str).str.replace('2 026', '2026').str.strip()
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime']).reset_index(drop=True)

pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
df['CO'] = pd.to_numeric(df['CO'], errors='coerce') * 1000  # 单位统一

# ==========================================
# 2. 特征工程 (滑动窗口前置)
# ==========================================
# 差分目标与滞后项
for p in pollutants:
    df[f'{p}_lag1'] = df[p].shift(1)
    df[f'{p}_delta'] = df[p] - df[f'{p}_lag1']

# 时间周期编码
df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)

# 气象滞后
meteo_cols = ['temperature', 'humidity']
for lag in [1, 6, 12]:
    for col in meteo_cols:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

df = df.dropna().reset_index(drop=True)

# 选定特征与目标
delta_cols = [f'{p}_delta' for p in pollutants]
lag1_cols = [f'{p}_lag1' for p in pollutants]
feature_cols = ['temperature', 'humidity', 'hour_sin', 'hour_cos'] + \
               [f'{c}_lag{l}' for l in [1,6,12] for c in meteo_cols] + lag1_cols

X_data = df[feature_cols]
Y_data = df[delta_cols]

# ==========================================
# 3. 序列重构 (LSTM 的灵魂：滑动窗口)
# ==========================================
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# 必须进行标准化，否则神经网络无法收敛
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_x.fit_transform(X_data)
Y_scaled = scaler_y.fit_transform(Y_data)

WINDOW_SIZE = 24  # 利用过去24小时预测下一小时
X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, WINDOW_SIZE)

# 划分训练/测试集 (80/20)
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
Y_train, Y_test = Y_seq[:split], Y_seq[split:]

# ==========================================
# 4. 堆叠 LSTM 模型构建
# ==========================================
print("🏗️ 正在构建双层堆叠 LSTM 网络...")
model = Sequential([
    Input(shape=(WINDOW_SIZE, X_train.shape[2])),
    LSTM(100, return_sequences=True, activation='tanh'),
    Dropout(0.2),
    LSTM(50, return_sequences=False, activation='tanh'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(6)  # 输出6种污染物的 ΔY
])

model.compile(optimizer='adam', loss='mse')

# 训练监控
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("🚀 开始炼丹 (LSTM 训练)...")
model.fit(X_train, Y_train, epochs=100, batch_size=64,
          validation_split=0.15, callbacks=[early_stop], verbose=1)

# ==========================================
# 5. 结果还原与协同性验证
# ==========================================
Y_pred_scaled = model.predict(X_test)
Y_pred_delta = scaler_y.inverse_transform(Y_pred_scaled)
Y_true_delta = scaler_y.inverse_transform(Y_test)

# 还原绝对浓度：Pred_t = Lag1_{t} + Delta_Pred_t
test_indices = df.index[split + WINDOW_SIZE:]
Y_test_real = df.loc[test_indices, pollutants].values
Y_test_lag1 = df.loc[test_indices, lag1_cols].values
Y_pred_real = Y_test_lag1 + Y_pred_delta

# 单位换算回 mg/m3 (针对 CO)
idx_co = pollutants.index('CO')
Y_test_real[:, idx_co] /= 1000
Y_pred_real[:, idx_co] /= 1000

# ==========================================
# 6. 输出对比图表
# ==========================================
# 1. 动态变化 (ΔY) 协同图
fig, ax = plt.subplots(1, 2, figsize=(16, 7))
f_dist = np.linalg.norm(pd.DataFrame(Y_true_delta).corr().values - pd.DataFrame(Y_pred_delta).corr().values)

sns.heatmap(pd.DataFrame(Y_true_delta, columns=pollutants).corr(), annot=True, cmap='coolwarm', ax=ax[0], fmt='.2f')
ax[0].set_title("真实变化量(ΔY)相关性")

sns.heatmap(pd.DataFrame(Y_pred_delta, columns=pollutants).corr(), annot=True, cmap='coolwarm', ax=ax[1], fmt='.2f')
ax[1].set_title(f"LSTM 预测变化量相关性\n(Frobenius 距离: {f_dist:.4f})")

plt.tight_layout()
plt.savefig('LSTM_Synergy_Final.png', dpi=300)
plt.show()

print("✅ LSTM 流程运行完毕，两套数据已对齐。")