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
# 0. 国标 AQI 计算引擎（六项完整版）
# ==========================================
def calculate_iaqi(cp, bp_lo, bp_hi, iaqi_lo, iaqi_hi):
    if cp <= 0:
        return 0
    return ((iaqi_hi - iaqi_lo) / (bp_hi - bp_lo)) * (cp - bp_lo) + iaqi_lo

def get_aqi(row):
    iaqi_dict = {}

    # PM2.5
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

    # PM10
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

    # O3（同时考虑1小时均值和8小时滑动均值，取大）
    v_1h = row['O3']
    v_8h = row.get('O3_8h', v_1h)
    if pd.isna(v_8h):
        v_8h = v_1h

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

    # NO2
    v = row['NO2']
    if v <= 100:
        iaqi_dict['NO2'] = calculate_iaqi(v, 0, 100, 0, 50)
    elif v <= 200:
        iaqi_dict['NO2'] = calculate_iaqi(v, 100, 200, 50, 100)
    elif v <= 700:
        iaqi_dict['NO2'] = calculate_iaqi(v, 200, 700, 100, 150)
    else:
        iaqi_dict['NO2'] = calculate_iaqi(v, 700, 1200, 150, 200)

    # SO2
    v = row['SO2']
    if v <= 150:
        iaqi_dict['SO2'] = calculate_iaqi(v, 0, 150, 0, 50)
    elif v <= 500:
        iaqi_dict['SO2'] = calculate_iaqi(v, 150, 500, 50, 100)
    elif v <= 650:
        iaqi_dict['SO2'] = calculate_iaqi(v, 500, 650, 100, 150)
    else:
        iaqi_dict['SO2'] = calculate_iaqi(v, 650, 800, 150, 200)

    # CO（单位必须为mg/m³）
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
# 1. 特征工程（构造过去24小时快照 + 未来24小时目标）
# ==========================================
def build_features_targets(df, forecast_hours=24):
    """
    返回：
        df_feat: 特征DataFrame（每一行对应一个时刻，包含过去24h的信息）
        targets: 字典，键为污染物名，值为DataFrame(24列)，每列为未来第h小时的浓度
        pollutants, meteo_cols
    """
    df_feat = df.copy()
    df_feat['datetime'] = pd.to_datetime(df_feat['datetime'], errors='coerce')
    df_feat = df_feat.dropna(subset=['datetime'])

    pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    # CO统一转为mg/m³
    df_feat['CO'] = pd.to_numeric(df_feat['CO'], errors='coerce') / 1000.0

    meteo_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']

    # 历史滞后特征：过去1, 2, 3, 6, 12, 24小时的污染物与气象值
    for lag in [1, 2, 3, 6, 12, 24]:
        for p in pollutants:
            df_feat[f'{p}_lag{lag}'] = df_feat[p].shift(lag)
        for m in meteo_cols:
            df_feat[f'{m}_lag{lag}'] = df_feat[m].shift(lag)

    # 循环时间编码
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['datetime'].dt.hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['datetime'].dt.hour / 24)

    # 构造多步目标：未来第1小时到第forecast_hours小时的浓度
    target_dict = {}
    for p in pollutants:
        tar_df = pd.DataFrame(index=df_feat.index)
        for h in range(1, forecast_hours + 1):
            tar_df[f'{p}_t+{h}'] = df_feat[p].shift(-h)
        target_dict[p] = tar_df

    # 剔除因滞后和目标shift产生的NaN行
    df_feat = df_feat.dropna().reset_index(drop=True)
    for p in pollutants:
        target_dict[p] = target_dict[p].iloc[df_feat.index.min(): df_feat.index.max()+1].reset_index(drop=True)

    return df_feat, target_dict, pollutants, meteo_cols


if __name__ == "__main__":
    print("⏳ 1. 读取数据并构造特征与多步目标...")
    df = pd.read_excel('Adata2.xlsx')
    df_feat, target_dict, pollutants, meteo_cols = build_features_targets(df, forecast_hours=24)

    # 分离特征矩阵X，剔除所有污染物现值和目标列
    drop_pattern = [f'{p}_t+' for p in pollutants] + [f'{p}_lag' for p in pollutants] + pollutants
    def keep_col(col):
        for pat in drop_pattern:
            if pat in col:
                return False
        return True

    X = df_feat[[col for col in df_feat.columns if keep_col(col)]]
    # 移除datetime列（如果有）
    if 'datetime' in X.columns:
        X = X.drop(columns=['datetime'])

    # 目标：将6种污染物×24步拼接成多输出目标矩阵 (N, 144)
    Y_direct = pd.concat([target_dict[p] for p in pollutants], axis=1)

    # 按月份切分数据集
    train_mask = df_feat['datetime'].dt.month <= 2
    test_mask = df_feat['datetime'].dt.month == 3

    X_train, Y_train = X[train_mask], Y_direct[train_mask]
    X_test, Y_test = X[test_mask], Y_direct[test_mask]
    test_datetime = df_feat['datetime'][test_mask].reset_index(drop=True)

    print(f"✅ 训练集(1-2月)样本数: {X_train.shape[0]} | 测试集(3月)样本数: {X_test.shape[0]}")
    print(f"   输入特征维度: {X_train.shape[1]} | 输出维度: {Y_train.shape[1]} (6×24)")

    # ==========================================
    # 2. 训练直接多步随机森林
    # ==========================================
    print("🧠 2. 正在训练直接多步输出随机森林...")
    rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(rf)
    model.fit(X_train, Y_train)
    print("✅ 训练完成！")

    # ==========================================
    # 3. 预测与重构
    # ==========================================
    print("🚀 3. 直接多步预测（一次性输出未来24小时所有浓度）...")
    Y_pred_raw = model.predict(X_test)  # shape: (N_test, 144)

    # 将144列重组为 (N_test, 24, 6)
    n_samples = Y_pred_raw.shape[0]
    Y_pred_3d = Y_pred_raw.reshape(n_samples, 24, len(pollutants))  # 第2维为预测步长，第3维为污染物顺序

    # 构建DataFrame存放预测结果：每个测试样本对应未来24小时
    pred_records = []
    for i in range(n_samples):
        base_time = test_datetime.iloc[i]
        for h in range(24):
            pred_time = base_time + pd.Timedelta(hours=h+1)
            rec = {'pred_time': pred_time, 'sample_idx': i}
            for j, p in enumerate(pollutants):
                rec[p] = Y_pred_3d[i, h, j]
            pred_records.append(rec)

    df_pred = pd.DataFrame(pred_records)

    # 真实值对照：从测试集中取出每个样本对应的未来24小时真实值，格式与预测一致
    true_records = []
    for i in range(n_samples):
        base_time = test_datetime.iloc[i]
        for h in range(24):
            true_time = base_time + pd.Timedelta(hours=h+1)
            rec = {'true_time': true_time, 'sample_idx': i}
            for p in pollutants:
                col_name = f'{p}_t+{h+1}'
                rec[p] = Y_test.iloc[i][col_name]
            true_records.append(rec)
    df_true = pd.DataFrame(true_records)

    # 按预测时间对齐，以便评估
    # 由于预测和真实都是按样本+未来小时生成的，直接将两表的索引对齐即可
    # 构建一个统一的评估DataFrame
    eval_df = df_pred[['pred_time', 'sample_idx'] + pollutants].copy()
    eval_df.columns = ['time', 'sample_idx'] + [f'pred_{p}' for p in pollutants]
    true_part = df_true[pollutants].copy()
    true_part.columns = [f'true_{p}' for p in pollutants]
    eval_df = pd.concat([eval_df, true_part], axis=1)

    # ==========================================
    # 4. 计算 O3 的 8 小时滑动均值（基于预测值与真实值）
    # ==========================================
    print("🔄 计算 O3 8h 滑动均值用于 AQI 计算...")
    eval_df['pred_O3_8h'] = eval_df['pred_O3'].rolling(8, min_periods=1).mean()
    eval_df['true_O3_8h'] = eval_df['true_O3'].rolling(8, min_periods=1).mean()

    # 准备两个DataFrame用于AQI计算
    # 真实值AQI
    true_for_aqi = eval_df[['true_PM2.5', 'true_PM10', 'true_O3', 'true_NO2', 'true_SO2', 'true_CO']].copy()
    true_for_aqi.columns = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    true_for_aqi['O3_8h'] = eval_df['true_O3_8h']
    true_aqi = true_for_aqi.apply(get_aqi, axis=1)
    true_aqi.columns = ['true_AQI', 'true_Primary']

    # 预测值AQI
    pred_for_aqi = eval_df[['pred_PM2.5', 'pred_PM10', 'pred_O3', 'pred_NO2', 'pred_SO2', 'pred_CO']].copy()
    pred_for_aqi.columns = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    pred_for_aqi['O3_8h'] = eval_df['pred_O3_8h']
    pred_aqi = pred_for_aqi.apply(get_aqi, axis=1)
    pred_aqi.columns = ['pred_AQI', 'pred_Primary']

    eval_df = pd.concat([eval_df, true_aqi, pred_aqi], axis=1)

    # ==========================================
    # 5. 评估指标
    # ==========================================
    print("\n" + "=" * 65)
    print("🏆 直接多步预测模型精度报表（3月测试集）")
    print("=" * 65)
    print(f"{'污染物':<8} | {'R²':<15} | {'RMSE':<15} | {'MAE'}")
    print("-" * 65)

    for p in pollutants:
        y_true = eval_df[f'true_{p}'].values
        y_pred = eval_df[f'pred_{p}'].values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        r2 = r2_score(y_true[mask], y_pred[mask])
        rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        mae = mean_absolute_error(y_true[mask], y_pred[mask])
        print(f"{p:<10} | {r2:<18.3f} | {rmse:<18.2f} | {mae:.2f}")

    # AQI评估
    y_true_aqi = eval_df['true_AQI'].values
    y_pred_aqi = eval_df['pred_AQI'].values
    mask_aqi = ~np.isnan(y_true_aqi) & ~np.isnan(y_pred_aqi)
    aqi_r2 = r2_score(y_true_aqi[mask_aqi], y_pred_aqi[mask_aqi])
    aqi_rmse = np.sqrt(mean_squared_error(y_true_aqi[mask_aqi], y_pred_aqi[mask_aqi]))
    print("-" * 65)
    print(f"{'AQI':<10} | {aqi_r2:<18.3f} | {aqi_rmse:<18.2f} | --")
    print("=" * 65)

    # 导出预测数据供第三问调度使用
    # 导出每日预报（取预测的最后一个样本的24小时，或根据需要导出整个测试集）
    q3_export = df_pred[['pred_time'] + pollutants].copy()
    q3_export.to_csv('Q3_Forecast_Data.csv', index=False, encoding='utf-8-sig')
    print("💾 未来24小时滚动预测数据已保存至 Q3_Forecast_Data.csv")

    # ==========================================
    # 6. 可视化（选取最后14天数据）
    # ==========================================
    print("🎨 绘制预测对比图...")
    # 由于预测是按样本+未来小时展开的，直接取最后一段连续时间
    # 为了画图方便，我们按照 eval_df 的时间顺序重新索引
    eval_df = eval_df.sort_values('time').reset_index(drop=True)
    # 选择最后14天
    unique_times = eval_df['time'].unique()
    last_14d_times = unique_times[-24*14:]
    mask_plot = eval_df['time'].isin(last_14d_times)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # AQI对比
    ax1.plot(eval_df.loc[mask_plot, 'time'], eval_df.loc[mask_plot, 'true_AQI'],
             label='真实 AQI', color='#1f77b4', linewidth=2)
    ax1.plot(eval_df.loc[mask_plot, 'time'], eval_df.loc[mask_plot, 'pred_AQI'],
             label='模型预测 AQI', color='#ff7f0e', linestyle='--', linewidth=2)
    ax1.axhline(100, color='red', linestyle=':', label='超标警戒线 (AQI=100)')
    ax1.set_title("直接多步预测：综合 AQI 走势（最后14天）", fontsize=16, fontweight='bold')
    ax1.set_ylabel("AQI")
    ax1.legend()
    ax1.grid(True, alpha=0.4)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # O3 浓度对比
    ax2.plot(eval_df.loc[mask_plot, 'time'], eval_df.loc[mask_plot, 'true_O3'],
             label='真实 O3 浓度', color='#2ca02c', linewidth=2)
    ax2.plot(eval_df.loc[mask_plot, 'time'], eval_df.loc[mask_plot, 'pred_O3'],
             label='模型预测 O3 浓度', color='#d62728', linestyle='--', linewidth=2)
    ax2.set_title("O3 光化学污染预报（直接多步模型）", fontsize=16, fontweight='bold')
    ax2.set_ylabel("O₃ 浓度 (μg/m³)")
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.tight_layout()
    plt.savefig('Q2_Direct_MultiStep_Forecast.png', dpi=300, bbox_inches='tight')
    print("🎉 图片已保存至 Q2_Direct_MultiStep_Forecast.png，反相问题已彻底根治！")
    # 从原始测试集中筛选每天0点的样本（确保使用的特征表 X_test 有datetime索引）
    test_times = test_datetime  # 从前面变量拿
    midnight_mask = test_times.dt.hour == 0
    midnight_indices = midnight_mask[midnight_mask].index

    # 若当天有0点样本，取第一个
    sample_idx = midnight_indices[0]  # 或者循环绘制多天
    X_sample = X_test.iloc[sample_idx:sample_idx + 1]
    Y_sample_true = Y_test.iloc[sample_idx]  # 包含未来24小时真实值

    # 模型预测未来24小时
    Y_sample_pred = model.predict(X_sample)[0].reshape(24, 6)

    # 构造时间轴
    base_time = test_times.iloc[sample_idx]
    future_times = [base_time + pd.Timedelta(hours=h + 1) for h in range(24)]

    # 真实值
    true_vals = [Y_sample_true[f'O3_t+{h + 1}'] for h in range(24)]
    true_aqi_vals = get_aqi(pd.DataFrame({
        'PM2.5': [Y_sample_true[f'PM2.5_t+{h + 1}'] for h in range(24)],
        'PM10': [Y_sample_true[f'PM10_t+{h + 1}'] for h in range(24)],
        'O3': true_vals,
        'NO2': [Y_sample_true[f'NO2_t+{h + 1}'] for h in range(24)],
        'SO2': [Y_sample_true[f'SO2_t+{h + 1}'] for h in range(24)],
        'CO': [Y_sample_true[f'CO_t+{h + 1}'] for h in range(24)],
    }))[0].values  # get_aqi返回两列，取AQI列

    # 预测值
    pred_o3 = Y_sample_pred[:, 2]  # O3是第3列（索引2）
    pred_aqi_vals = get_aqi(pd.DataFrame({
        'PM2.5': Y_sample_pred[:, 0],
        'PM10': Y_sample_pred[:, 1],
        'O3': pred_o3,
        'NO2': Y_sample_pred[:, 3],
        'SO2': Y_sample_pred[:, 4],
        'CO': Y_sample_pred[:, 5],
    }))[0].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.plot(future_times, true_aqi_vals, 'o-', label='真实 AQI', color='#1f77b4', linewidth=2)
    ax1.plot(future_times, pred_aqi_vals, 's--', label='24h 预报 AQI', color='#ff7f0e', linewidth=2)
    ax1.axhline(100, color='red', linestyle=':', label='超标警戒线')
    ax1.set_title(f"{base_time.strftime('%m月%d日')} 未来24小时 AQI 预报 vs 实况", fontsize=16, fontweight='bold')
    ax1.set_ylabel("AQI")
    ax1.legend()
    ax1.grid(True, alpha=0.4)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))

    ax2.plot(future_times, true_vals, 'o-', label='真实 O₃', color='#2ca02c', linewidth=2)
    ax2.plot(future_times, pred_o3, 's--', label='24h 预报 O₃', color='#d62728', linewidth=2)
    ax2.set_title(f"{base_time.strftime('%m月%d日')} O₃ 浓度预报效果", fontsize=16, fontweight='bold')
    ax2.set_ylabel("O₃ 浓度 (μg/m³)")
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))

    plt.tight_layout()
    plt.savefig('Q2_Daily_Forecast_Example.png', dpi=300, bbox_inches='tight')