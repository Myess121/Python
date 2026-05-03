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

# 基础设置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据加载与预处理 (适配全新 Adata2 数据集)
# ==========================================
print("📂 正在加载并清洗新数据集 Adata2...")
# 请确保文件名和后缀对应，如果是 csv 请用 pd.read_csv('Adata2.csv')
file_path = 'Adata2.xlsx'

# 新数据表头很规整，不需要 skiprows=9 了
df = pd.read_excel(file_path)

# 时间格式很标准，直接转换即可
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime']).reset_index(drop=True)

pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
df['CO'] = pd.to_numeric(df['CO'], errors='coerce') * 1000  # 统一为 ug/m3

# 特征工程：构建浓度差分项
for p in pollutants:
    df[f'{p}_lag1'] = df[p].shift(1)
    df[f'{p}_delta'] = df[p] - df[f'{p}_lag1']

# 时间周期编码
df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)

# 🌟 核心升级：将新发现的风速、风向、气压加入气象特征库！
meteo_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']
for lag in [1, 6, 12]:
    for col in meteo_cols:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

df = df.dropna().reset_index(drop=True)

# 准备最终喂给模型的特征
delta_cols = [f'{p}_delta' for p in pollutants]
lag1_cols = [f'{p}_lag1' for p in pollutants]

# 组合所有特征
feature_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'hour_sin', 'hour_cos'] + \
               [f'{c}_lag{l}' for l in [1, 6, 12] for c in meteo_cols] + lag1_cols

X_data = df[feature_cols]
Y_data = df[delta_cols]

# ==========================================
# 2. LSTM 序列重构与归一化
# ==========================================
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_x.fit_transform(X_data)
Y_scaled = scaler_y.fit_transform(Y_data)


def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


WINDOW_SIZE = 24
X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, WINDOW_SIZE)

split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
Y_train, Y_test = Y_seq[:split], Y_seq[split:]

# ==========================================
# 3. 建立并训练 LSTM
# ==========================================
print("🏗️ 训练堆叠 LSTM 模型...")
model = Sequential([
    Input(shape=(WINDOW_SIZE, X_train.shape[2])),
    LSTM(100, return_sequences=True, activation='tanh'),
    Dropout(0.2),
    LSTM(50, activation='tanh'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(6)
])
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_split=0.15, callbacks=[early_stop], verbose=0)

# ==========================================
# 4. 预测还原与评估 (按照你要求的格式)
# ==========================================
Y_pred_s = model.predict(X_test)
Y_pred_delta = scaler_y.inverse_transform(Y_pred_s)
Y_true_delta = scaler_y.inverse_transform(Y_test)

# 还原绝对浓度用于 R2 计算
test_indices = df.index[split + WINDOW_SIZE:]
Y_test_real = df.loc[test_indices, pollutants].values
Y_test_lag1 = df.loc[test_indices, lag1_cols].values
Y_pred_real = Y_test_lag1 + Y_pred_delta

# 打印 🏆 模型竞技场表格
print("\n🏆 模型竞技场对比 (持久性基准 Baseline VS 本文 LSTM 模型)")
print("=" * 90)
for i, p in enumerate(pollutants):
    y_t = Y_test_real[:, i]
    y_p = Y_pred_real[:, i]
    y_b = Y_test_lag1[:, i]  # 基准：上一小时浓度

    # 针对 CO 恢复单位展示
    factor = 1000 if p == 'CO' else 1
    r2_m = r2_score(y_t, y_p)
    r2_b = r2_score(y_t, y_b)
    rmse_m = np.sqrt(mean_squared_error(y_t, y_p)) / factor
    rmse_b = np.sqrt(mean_squared_error(y_t, y_b)) / factor
    imp = (rmse_b - rmse_m) / rmse_b * 100

    p_label = f"[{p:<5}]"
    print(
        f"{p_label} | 本文 R2: {r2_m:.3f} (基准: {r2_b:.3f}) | 本文 RMSE: {rmse_m:.2f} (基准: {rmse_b:.2f}) -> 误差降低: {imp:.1f}%")
print("=" * 90)

# ==========================================
# 5. 协同矩阵计算与绘图
# ==========================================
# 1. 绝对浓度协同
corr_true = pd.DataFrame(Y_test_real, columns=pollutants).corr()
corr_pred = pd.DataFrame(Y_pred_real, columns=pollutants).corr()
frob_abs = np.linalg.norm(corr_true.values - corr_pred.values)
print(f"\n【协同矩阵指标】真实与预测相关性矩阵的 Frobenius 范数距离: {frob_abs:.4f}")

fig1, ax1 = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(corr_true, annot=True, cmap='coolwarm', ax=ax1[0], fmt='.2f')
ax1[0].set_title("实测浓度：多污染物协同矩阵")
sns.heatmap(corr_pred, annot=True, cmap='coolwarm', ax=ax1[1], fmt='.2f')
ax1[1].set_title(f"LSTM 预测浓度协同矩阵 (F-dist: {frob_abs:.4f})")
plt.savefig('Q2_Synergy_Correlation.png', dpi=300)

# 2. 动态变化 ΔY 协同
corr_d_true = pd.DataFrame(Y_true_delta, columns=pollutants).corr()
corr_d_pred = pd.DataFrame(Y_pred_delta, columns=pollutants).corr()
frob_delta = np.linalg.norm(corr_d_true.values - corr_d_pred.values)
print(f"\n🔄 动态协同验证 (预测变化量 ΔY 的相关性):")
print(f"ΔY 协同矩阵 Frobenius 距离: {frob_delta:.4f}")

fig2, ax2 = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(corr_d_true, annot=True, cmap='coolwarm', ax=ax2[0], fmt='.2f')
ax2[0].set_title("真实变化量(ΔY)协同矩阵")
sns.heatmap(corr_d_pred, annot=True, cmap='coolwarm', ax=ax2[1], fmt='.2f')
ax2[1].set_title(f"模型预测变化量(ΔY)协同矩阵 (F-dist: {frob_delta:.4f})")
plt.savefig('Q2_Synergy_Delta_Correlation.png', dpi=300)

print("\n✅ 全部完成！图片已保存为 'Q2_Synergy_Correlation.png' 和 'Q2_Synergy_Delta_Correlation.png'")

# --- 在原有代码末尾追加 ---

# 1. 整理数据
results_data = []
for i, p in enumerate(pollutants):
    y_t, y_p, y_b = Y_test_real[:, i], Y_pred_real[:, i], Y_test_lag1[:, i]
    factor = 1000 if p == 'CO' else 1

    results_data.append({
        '污染物': p,
        '本文 R2': r2_score(y_t, y_p),
        '基准 R2': r2_score(y_t, y_b),
        '本文 RMSE': np.sqrt(mean_squared_error(y_t, y_p)) / factor,
        '基准 RMSE': np.sqrt(mean_squared_error(y_t, y_b)) / factor,
        '误差降低率(%)': ((np.sqrt(mean_squared_error(y_t, y_b)) - np.sqrt(mean_squared_error(y_t, y_p))) / np.sqrt(
            mean_squared_error(y_t, y_b))) * 100
    })

# 2. 保存为 CSV
df_final_results = pd.DataFrame(results_data)
df_final_results.to_csv('LSTM_Comparison_Results.csv', index=False, encoding='utf-8-sig')

print("\n📊 结果已自动保存至 'LSTM_Comparison_Results.csv'，可以直接用 Excel 打开填表！")