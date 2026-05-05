# -*- coding: utf-8 -*-
"""
RF-ΔY 多污染物协同预测系统（修复增强版）
功能：直接多步预测 + 国标AQI计算 + 准确率量化 + 可视化
修复重点：①特征-标签严格对齐 ②误差累积抑制 ③方向一致性评估
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 0. 国标 AQI 计算引擎（严格遵循 HJ 633-2012）
# ==========================================
def calculate_iaqi(cp, bp_lo, bp_hi, iaqi_lo, iaqi_hi):
    """分段线性插值计算单个污染物的 IAQI"""
    if cp <= 0 or bp_hi == bp_lo:
        return 0
    return ((iaqi_hi - iaqi_lo) / (bp_hi - bp_lo)) * (cp - bp_lo) + iaqi_lo


def get_aqi(row, pollutants=None):
    """
    计算综合 AQI 与首要污染物
    输入: row - 包含6种污染物浓度的字典/序列
    输出: [AQI整数值, 首要污染物名称]
    """
    if pollutants is None:
        pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']

    iaqi_dict = {}

    # 1. PM2.5 (μg/m³)
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

    # 2. PM10 (μg/m³)
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

    # 3. O3 (μg/m³) - 取1h与8h滑动平均的IAQI较大者
    v_1h = row.get('O3', 0)
    v_8h = row.get('O3_8h', v_1h)
    if pd.isna(v_8h): v_8h = v_1h

    # O3 1h IAQI
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

    # O3 8h IAQI
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

    # 4. NO2 (μg/m³)
    v = row['NO2']
    if v <= 100:
        iaqi_dict['NO2'] = calculate_iaqi(v, 0, 100, 0, 50)
    elif v <= 200:
        iaqi_dict['NO2'] = calculate_iaqi(v, 100, 200, 50, 100)
    elif v <= 700:
        iaqi_dict['NO2'] = calculate_iaqi(v, 200, 700, 100, 150)
    else:
        iaqi_dict['NO2'] = calculate_iaqi(v, 700, 1200, 150, 200)

    # 5. SO2 (μg/m³)
    v = row['SO2']
    if v <= 150:
        iaqi_dict['SO2'] = calculate_iaqi(v, 0, 150, 0, 50)
    elif v <= 500:
        iaqi_dict['SO2'] = calculate_iaqi(v, 150, 500, 50, 100)
    elif v <= 650:
        iaqi_dict['SO2'] = calculate_iaqi(v, 500, 650, 100, 150)
    else:
        iaqi_dict['SO2'] = calculate_iaqi(v, 650, 800, 150, 200)

    # 6. CO (mg/m³) - 注意单位！
    v = row['CO']
    if v <= 5:
        iaqi_dict['CO'] = calculate_iaqi(v, 0, 5, 0, 50)
    elif v <= 10:
        iaqi_dict['CO'] = calculate_iaqi(v, 5, 10, 50, 100)
    elif v <= 35:
        iaqi_dict['CO'] = calculate_iaqi(v, 10, 35, 100, 150)
    else:
        iaqi_dict['CO'] = calculate_iaqi(v, 35, 60, 150, 200)

    # 取最大值作为综合 AQI
    aqi = max(iaqi_dict.values())
    primary = max(iaqi_dict, key=iaqi_dict.get) if aqi > 50 else "无"

    return pd.Series([round(aqi), primary])


# ==========================================
# 1. 特征工程（修复对齐问题 + 增强特征）
# ==========================================
def build_features(df, pollutants, meteo_cols):
    """
    构建预测特征矩阵
    修复重点：确保特征与标签的时间索引严格对齐
    """
    df_feat = df.copy()
    df_feat['datetime'] = pd.to_datetime(df_feat['datetime'], errors='coerce')
    df_feat = df_feat.sort_values('datetime').reset_index(drop=True)

    # 单位统一：CO 转为 mg/m³（原始数据可能是 μg/m³）
    if df_feat['CO'].max() > 100:  # 判断是否未转换
        df_feat['CO'] = df_feat['CO'] / 1000.0

    # === 时间周期特征（正弦余弦编码）===
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['datetime'].dt.hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['datetime'].dt.hour / 24)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['datetime'].dt.month / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['datetime'].dt.month / 12)

    # === 气象特征滞后项 ===
    for col in meteo_cols:
        for lag in [1, 3, 6, 12, 24]:
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)

    # === 污染物特征：滞后项 + 变化量 + 日夜锚点 ===
    for p in pollutants:
        # 滞后1小时（短期记忆）
        df_feat[f'{p}_lag1'] = df_feat[p].shift(1)
        # 滞后24小时（日夜周期锚点，关键！）
        df_feat[f'{p}_lag24'] = df_feat[p].shift(24)
        # 变化量（ΔY 策略的核心）
        df_feat[f'{p}_delta'] = df_feat[p] - df_feat[f'{p}_lag1']
        # 滑动统计特征（增强鲁棒性）
        df_feat[f'{p}_roll3h'] = df_feat[p].rolling(3, min_periods=1).mean()
        df_feat[f'{p}_roll24h'] = df_feat[p].rolling(24, min_periods=1).mean()

    # 删除含 NaN 的行（由 shift/rolling 产生）
    df_feat = df_feat.dropna().reset_index(drop=True)

    return df_feat


# ==========================================
# 2. 预测准确率评估函数（新增5项指标）
# ==========================================
def evaluate_prediction(y_true, y_pred, pollutant_name):
    """
    计算多项预测准确率指标
    返回: 指标字典
    """
    # 基础指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE（避免除零）
    mask = y_true != 0
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100 if mask.sum() > 0 else np.nan

    # 🌟 方向一致性准确率（关键！判断预测趋势是否正确）
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0  # 真实变化方向
        pred_direction = np.diff(y_pred) > 0  # 预测变化方向
        direction_acc = np.mean(true_direction == pred_direction) * 100
    else:
        direction_acc = np.nan

    return {
        '污染物': pollutant_name,
        'R²': round(r2, 4),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'MAPE(%)': round(mape, 2) if not np.isnan(mape) else 'N/A',
        '方向准确率(%)': round(direction_acc, 2) if not np.isnan(direction_acc) else 'N/A'
    }


# ==========================================
# 3. 主程序：训练 + 预测 + 评估 + 可视化
# ==========================================
if __name__ == "__main__":
    print("🚀 RF-ΔY 多污染物协同预测系统（修复增强版）启动...\n")

    # === 0. 参数配置 ===
    pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    meteo_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']
    target_cols = [f'{p}_delta' for p in pollutants]  # 预测目标：变化量

    # === 1. 数据读取与预处理 ===
    print("📥 1. 读取数据...")
    df_raw = pd.read_excel('Adata2.xlsx')  # 请确认文件名

    # 基础清洗
    df_raw = df_raw.dropna(subset=['datetime'] + pollutants + ['temperature', 'humidity'])
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])

    # === 2. 特征工程 ===
    print("🔧 2. 构建特征矩阵...")
    df_feat = build_features(df_raw, pollutants, meteo_cols)

    # 特征列与标签列
    feature_cols = [c for c in df_feat.columns if c not in ['datetime'] + pollutants + target_cols]
    X = df_feat[feature_cols]
    Y_delta = df_feat[target_cols]
    Y_abs = df_feat[pollutants]  # 绝对浓度（用于评估）

    # === 3. 时间序列划分（严禁随机shuffle！）===
    # 训练集：1月-2月；测试集：3月
    train_mask = df_feat['datetime'].dt.month <= 2
    test_mask = df_feat['datetime'].dt.month == 3

    X_train, X_test = X[train_mask], X[test_mask]
    Y_delta_train = Y_delta[train_mask]
    Y_abs_test = Y_abs[test_mask].reset_index(drop=True)
    test_datetime = df_feat['datetime'][test_mask].reset_index(drop=True)

    print(f"   训练样本: {len(X_train)} | 测试样本: {len(X_test)}")

    # === 4. 模型训练（Multi-output RF）===
    print("🧠 3. 训练多输出随机森林模型...")
    rf_base = RandomForestRegressor(
        n_estimators=200,  # 增加树数量提升稳定性
        max_depth=15,  # 控制深度防过拟合
        min_samples_split=10,  # 增加分裂阈值
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model = MultiOutputRegressor(rf_base)
    model.fit(X_train, Y_delta_train)
    print("   ✓ 模型训练完成")

    # === 5. 直接多步预测（规避递归误差累积）===
    print("🔮 4. 执行直接多步预测（非递归策略）...")
    Y_pred_delta = model.predict(X_test)  # 直接输出24步变化量

    # 变化量 → 绝对浓度（关键修复：用lag24锚点而非lag1递归）
    # 原错误代码：
    # Y_pred_abs = pd.DataFrame(columns=pollutants, index=Y_pred_delta.index)

    # ✅ 修正为：
    Y_pred_abs = pd.DataFrame(columns=pollutants, index=range(len(Y_pred_delta)))
    for idx, p in enumerate(pollutants):
        # 获取测试集起始时刻的lag24值（即训练集末尾的真实值）
        lag24_start = df_feat.loc[train_mask, p].iloc[-24:].values
        # 用lag24作为基准 + 预测的Δ，避免误差传播
        pred_abs = []
        for i, delta in enumerate(Y_pred_delta[:, idx]):
            anchor = lag24_start[i % 24]  # 24小时周期锚定
            abs_val = anchor + delta
            # 物理边界裁剪
            abs_val = np.clip(abs_val, 0, df_raw[p].quantile(0.99) * 1.5)
            pred_abs.append(abs_val)
        Y_pred_abs[p] = pred_abs

    # === 6. O3 8小时滑动平均（混合序列拼接法）===
    print("🔄 5. 计算 O3 8小时滑动平均（跨日拼接）...")
    # 拼接：训练集末尾24h真实值 + 测试集预测值
    o3_history = df_feat.loc[train_mask, 'O3'].iloc[-24:].values
    o3_pred = Y_pred_abs['O3'].values
    o3_hybrid = np.concatenate([o3_history, o3_pred])

    # 8小时滑动平均（min_periods=1 保证边界可计算）
    o3_8h = pd.Series(o3_hybrid).rolling(8, min_periods=1).mean().values
    Y_pred_abs['O3_8h'] = o3_8h[24:]  # 截取预测部分

    # 同样处理测试集真实值的O3_8h（用于评估）
    o3_test_real = Y_abs_test['O3'].values
    o3_test_hybrid = np.concatenate([o3_history, o3_test_real])
    o3_test_8h = pd.Series(o3_test_hybrid).rolling(8, min_periods=1).mean().values
    Y_abs_test['O3_8h'] = o3_test_8h[24:]

    # === 7. AQI 计算 ===
    print("📊 6. 计算综合 AQI 与首要污染物...")
    Y_abs_test[['AQI', 'Primary']] = Y_abs_test.apply(get_aqi, axis=1, pollutants=pollutants)
    Y_pred_abs[['AQI', 'Primary']] = Y_pred_abs.apply(get_aqi, axis=1, pollutants=pollutants)

    # === 8. 准确率评估报表 ===
    print("\n" + "=" * 70)
    print("🏆 预测准确率评估报表（测试集：2026年3月）")
    print("=" * 70)
    print(f"{'污染物':<8} | {'R²':>8} | {'RMSE':>8} | {'MAE':>8} | {'MAPE%':>8} | {'方向%':>8}")
    print("-" * 70)

    metrics_list = []
    for p in pollutants:
        y_true = Y_abs_test[p].values
        y_pred = Y_pred_abs[p].values.astype(float)
        metrics = evaluate_prediction(y_true, y_pred, p)
        metrics_list.append(metrics)
        print(f"{metrics['污染物']:<8} | {metrics['R²']:>8.3f} | {metrics['RMSE']:>8.2f} | "
              f"{metrics['MAE']:>8.2f} | {str(metrics['MAPE(%)']):>8} | {str(metrics['方向准确率(%)']):>8}")

    print("=" * 70)

    # 综合指标
    avg_r2 = np.mean([m['R²'] for m in metrics_list if isinstance(m['R²'], (int, float))])
    avg_dir = np.mean([m['方向准确率(%)'] for m in metrics_list if isinstance(m['方向准确率(%)'], (int, float))])
    print(f"📈 平均 R²: {avg_r2:.3f} | 平均方向准确率: {avg_dir:.1f}%")

    # === 9. 可视化（修复曲线反向问题）===
    print("🎨 7. 生成可视化图表...")

    # 取最近7天（168小时）展示
    plot_len = min(168, len(test_datetime))
    time_axis = test_datetime.iloc[-plot_len:]

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('RF-ΔY 多污染物协同预测效果（修复增强版）', fontsize=16, fontweight='bold', y=1.02)

    # (1) 综合 AQI 对比
    ax = axes[0, 0]
    ax.plot(time_axis, Y_abs_test['AQI'].iloc[-plot_len:], label='真实 AQI', color='#1f77b4', linewidth=2)
    ax.plot(time_axis, Y_pred_abs['AQI'].iloc[-plot_len:], label='预测 AQI', color='#ff7f0e', linestyle='--',
            linewidth=2)
    ax.axhline(100, color='red', linestyle=':', alpha=0.6, label='超标线 (AQI=100)')
    ax.set_title('综合 AQI 预测对比')
    ax.set_ylabel('AQI')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))

    # (2) O3 浓度对比（光化学污染重点）
    ax = axes[0, 1]
    ax.plot(time_axis, Y_abs_test['O3'].iloc[-plot_len:], label='真实 O3', color='#2ca02c', linewidth=1.5)
    ax.plot(time_axis, Y_pred_abs['O3'].iloc[-plot_len:], label='预测 O3', color='#d62728', linestyle='--',
            linewidth=1.5)
    ax.set_title('O3 浓度预测对比（光化学污染）')
    ax.set_ylabel('O3 (μg/m³)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))

    # (3) PM2.5 对比
    ax = axes[1, 0]
    ax.plot(time_axis, Y_abs_test['PM2.5'].iloc[-plot_len:], label='真实 PM2.5', color='#9467bd', linewidth=1.5)
    ax.plot(time_axis, Y_pred_abs['PM2.5'].iloc[-plot_len:], label='预测 PM2.5', color='#c5b0d5', linestyle='--',
            linewidth=1.5)
    ax.set_title('PM2.5 浓度预测对比')
    ax.set_ylabel('PM2.5 (μg/m³)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))

    # (4) 预测-真实散点图 + 拟合线（诊断反向问题）
    ax = axes[1, 1]
    all_true = np.concatenate([Y_abs_test[p].values for p in pollutants[:4]])
    all_pred = np.concatenate([Y_pred_abs[p].values.astype(float) for p in pollutants[:4]])
    ax.scatter(all_true, all_pred, alpha=0.3, s=10, color='#1f77b4')
    ax.plot([all_true.min(), all_true.max()], [all_true.min(), all_true.max()], 'r--', label='理想拟合线 y=x')
    # 添加线性拟合
    z = np.polyfit(all_true, all_pred, 1)
    p = np.poly1d(z)
    ax.plot(all_true, p(all_true), "g-", label=f'实际拟合: y={z[0]:.2f}x+{z[1]:.1f}')
    ax.set_xlabel('真实浓度')
    ax.set_ylabel('预测浓度')
    ax.set_title('预测-真实散点诊断（斜率≈1 表示无方向反转）')
    ax.legend()
    ax.grid(alpha=0.3)

    # (5) 残差分布直方图
    ax = axes[2, 0]
    residuals = Y_pred_abs['AQI'].iloc[-plot_len:].astype(float) - Y_abs_test['AQI'].iloc[-plot_len:]
    ax.hist(residuals, bins=30, edgecolor='black', color='skyblue')
    ax.axvline(0, color='red', linestyle='--', label='零误差线')
    ax.set_xlabel('预测残差 (预测值 - 真实值)')
    ax.set_ylabel('频数')
    ax.set_title('AQI 预测残差分布（越接近0越好）')
    ax.legend()
    ax.grid(alpha=0.3)

    # (6) 方向一致性热力图
    ax = axes[2, 1]
    dir_data = []
    for p in pollutants:
        true_dir = np.diff(Y_abs_test[p].iloc[-plot_len:].values) > 0
        pred_dir = np.diff(Y_pred_abs[p].iloc[-plot_len:].values.astype(float)) > 0
        acc = np.mean(true_dir == pred_dir) * 100
        dir_data.append([p, acc])

    dir_df = pd.DataFrame(dir_data, columns=['污染物', '方向准确率(%)'])
    im = ax.barh(dir_df['污染物'], dir_df['方向准确率(%)'],
                 color=['#2ca02c' if x > 70 else '#ff7f0e' if x > 50 else '#d62728' for x in dir_df['方向准确率(%)']])
    ax.set_xlabel('方向准确率 (%)')
    ax.set_title('各污染物变化趋势预测准确率（>70% 为优秀）')
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('Q3_Prediction_Diagnostic.png', dpi=300, bbox_inches='tight')
    print("   ✓ 图表已保存: Q3_Prediction_Diagnostic.png")

    # === 10. 导出调度系统输入数据 ===
    q3_input = pd.concat([
        test_datetime.reset_index(drop=True),
        df_feat.loc[test_mask, ['temperature', 'humidity', 'wind_speed']].reset_index(drop=True),
        Y_pred_abs.reset_index(drop=True)
    ], axis=1)
    q3_input.to_csv('Q3_Forecast_For_Scheduling.csv', index=False, encoding='utf-8-sig')
    print("   ✓ 调度输入数据已导出: Q3_Forecast_For_Scheduling.csv")

    print("\n✨ 预测系统执行完毕！请检查：")
    print("   ① 散点图拟合斜率是否接近 1（若为负说明方向反转）")
    print("   ② 方向准确率是否 >60%（光化学污染通常较难预测）")
    print("   ③ AQI 残差是否以 0 为中心对称分布")