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
# 2. 数据加载与 MIMO 特征准备
# ==========================================
print("📂 正在加载数据并构建 MIMO 宏观特征...")
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

# 基础特征列：不用做滞后 shift，因为我们要一次性截取 24 小时的数据块
feature_cols = pollutants + meteo_cols + ['hour_sin', 'hour_cos']

# ==========================================
# 3. 提取 1-2 月数据，构建 MIMO 训练集
# ==========================================
print("🏗️ 准备训练端到端(End-to-End)的随机森林模型...")
# 🌟 战术变更 1：训练集扩展到 3 月 15 日，让模型学会春季的爆发规律！
df_train = df[df['datetime'] < '2026-03-16 00:00:00'].copy()

LOOK_BACK = 24  # 观察过去 24 小时
HORIZON = 24  # 直接输出未来 24 小时


def create_mimo_dataset(data_df, lookback, horizon):
    X, Y = [], []
    for i in range(len(data_df) - lookback - horizon + 1):
        # 截取过去 24 小时的所有特征，并展平成一维长向量 (24 * 13 = 312 个特征)
        x_window = data_df[feature_cols].iloc[i: i + lookback].values.flatten()
        # 截取未来 24 小时的六项污染物，展平为目标向量 (24 * 6 = 144 个预测目标)
        y_window = data_df[pollutants].iloc[i + lookback: i + lookback + horizon].values.flatten()

        X.append(x_window)
        Y.append(y_window)
    return np.array(X), np.array(Y)


X_train, Y_train = create_mimo_dataset(df_train, LOOK_BACK, HORIZON)

# 训练随机森林（无需 StandardScaler，树模型万岁！）
print("🚀 开始训练随机森林 (MIMO架构, 自动并行计算中)...")
rf_base = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
# MultiOutputRegressor 让 RF 能够同时预测 144 个目标
model = MultiOutputRegressor(rf_base)
model.fit(X_train, Y_train)

# ==========================================
# 4. 第三问：三月份每日 22:00 “一键预测”
# ==========================================
print("\n🔄 开始三月份业务级仿真 (每日 22:00 一键下发次日 24H 预报)...")
# 🌟 战术变更 2：验证期改为 3 月 16 日至 31 日 (预测最后半个月)
predict_days = range(16, 32) # 16 到 31
all_predictions = []

for day in predict_days:
    # 锚点：每天晚上 22:00 (比如 day=16，锚点就是 3月15日 22:00)
    anchor_time = pd.to_datetime(f'2026-03-{day - 1:02d} 22:00:00')

    window_start = anchor_time + pd.Timedelta(hours=1)  # 预测起点：今晚 23:00
    window_end = anchor_time + pd.Timedelta(hours=24)  # 预测终点：明晚 22:00

    if window_end > df['datetime'].max():
        continue

    # 提取截止到今晚 22:00 的过去 24 小时真实历史数据
    history_start = anchor_time - pd.Timedelta(hours=LOOK_BACK - 1)
    df_history = df[(df['datetime'] >= history_start) & (df['datetime'] <= anchor_time)].copy()

    # 拦截无效数据
    if len(df_history) < LOOK_BACK or df_history[feature_cols].isnull().values.any():
        continue

    # 构建当前的一维长特征向量
    X_current = df_history[feature_cols].values.flatten().reshape(1, -1)

    # 🌟 见证奇迹：一次 predict 解决战斗！没有滚动！没有累加误差！
    y_pred_flat = model.predict(X_current)[0]

    # 把展平的 (144,) 结果重新折叠成 (24小时, 6项污染物) 的矩阵
    y_pred_matrix = y_pred_flat.reshape((HORIZON, len(pollutants)))

    future_times = pd.date_range(window_start, window_end, freq='h')

    for h_idx, target_time in enumerate(future_times):
        row_result = {'datetime': target_time}
        for p_idx, p in enumerate(pollutants):
            val = max(0, y_pred_matrix[h_idx][p_idx])  # 保证浓度不为负
            row_result[f'{p}_pred'] = val
        all_predictions.append(row_result)

    if day % 5 == 0 or day == 1:
        print(f"✅ 成功发布 {window_end.strftime('%Y-%m-%d')} 的全天预报！")

# ==========================================
# 5. 整理指标大表与画图
# ==========================================
df_pred = pd.DataFrame(all_predictions)
# 🌟 战术变更 3：评价基准也要对齐到 3月16日-31日
df_true = df[df['datetime'] >= '2026-03-16 00:00:00'][['datetime'] + pollutants].copy()
df_eval = pd.merge(df_true, df_pred, on='datetime', how='inner')

print("\n🏆 RF-MIMO 架构三月全月预测评价指标")
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

pd.DataFrame(eval_results).to_csv('Q3_March_RF_MIMO_Metrics.csv', index=False, encoding='utf-8-sig')

df_eval['True_AQI'] = df_eval.apply(get_comprehensive_aqi, axis=1)
temp_pred_df = df_eval[['datetime'] + [f'{p}_pred' for p in pollutants]].rename(
    columns={f'{p}_pred': p for p in pollutants})
df_eval['Pred_AQI'] = temp_pred_df.apply(get_comprehensive_aqi, axis=1)
df_eval['活动建议'] = df_eval['Pred_AQI'].apply(get_aqi_level)
df_eval.to_csv('Q3_March_RF_MIMO_Predictions.csv', index=False, encoding='utf-8-sig')

# ============ 画图验证 ============
print("\n📊 正在生成最后 5 天 MIMO 预测对比图...")
df_plot = df_eval[df_eval['datetime'] >= '2026-03-27 00:00:00'].set_index('datetime')

# 图 1：AQI 趋势
plt.figure(figsize=(14, 5))
plt.plot(df_plot.index, df_plot['True_AQI'], label='真实 AQI', color='black', marker='o', markersize=4)
plt.plot(df_plot.index, df_plot['Pred_AQI'], label='RF(MIMO)预测 AQI', color='#FF5722', linestyle='--', marker='^',
         markersize=4)
plt.axhline(y=100, color='red', linestyle=':', label='污染警戒线 (AQI=100)')
plt.fill_between(df_plot.index, df_plot['True_AQI'], df_plot['Pred_AQI'], color='gray', alpha=0.1)
plt.title('基于 RF-MIMO 架构的良乡校区空气质量(AQI) 直接多步预测', fontsize=16)
plt.ylabel('AQI 指数')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q3_RF_MIMO_Last5Days_AQI.png', dpi=300)

# 图 2：六项污染物矩阵
fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
axes = axes.flatten()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, p in enumerate(pollutants):
    ax = axes[i]
    ax.plot(df_plot.index, df_plot[p], label='真实值', color='black', linewidth=1.5)
    ax.plot(df_plot.index, df_plot[f'{p}_pred'], label='预测值', color=colors[i], linestyle='--', linewidth=2)
    unit = 'mg/m³' if p == 'CO' else 'μg/m³'
    ax.set_title(f'{p} MIMO 直接映射预测', fontweight='bold')
    ax.set_ylabel(f'浓度 ({unit})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('最后5天六项污染物协同预测 (基于“一天映射一天”策略)', fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig('Q3_RF_MIMO_Last5Days_Pollutants.png', dpi=300)

print("✅ 大功告成！不仅快，而且稳！快去看看生成的神仙图表吧！")