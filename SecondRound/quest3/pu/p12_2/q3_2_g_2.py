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


# ---------------- 1. 特征工程与基础函数 (继承第二问并优化) ----------------
# ---------------- 1. 修复后的特征工程 ----------------
def build_features(df, is_predict=False):
    df_feat = df.copy()
    df_feat['datetime'] = pd.to_datetime(df_feat['datetime'], errors='coerce')
    df_feat = df_feat.dropna(subset=['datetime']).sort_values('datetime')

    meteo_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']

    # 构建气象滞后特征
    temp_lags = [1, 6, 12, 13, 14]
    for lag in temp_lags:
        df_feat[f'Temp_lag{lag}'] = df_feat['temperature'].shift(lag)

    hum_lags = [1, 13, 24]
    for lag in hum_lags:
        df_feat[f'Hum_lag{lag}'] = df_feat['humidity'].shift(lag)

    new_meteo_cols = ['wind_speed', 'wind_direction', 'pressure']
    for lag in [1, 6, 12]:
        for col in new_meteo_cols:
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)

    # 时间周期编码
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['datetime'].dt.hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['datetime'].dt.hour / 24)

    # 污染物差分处理
    pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    df_feat['CO'] = pd.to_numeric(df_feat['CO'], errors='coerce') * 1000

    for p in pollutants:
        if p != 'CO':
            df_feat[p] = pd.to_numeric(df_feat[p], errors='coerce')
        df_feat[f'{p}_lag1'] = df_feat[p].shift(1)
        df_feat[f'{p}_delta'] = df_feat[p] - df_feat[f'{p}_lag1']

    # 🌟 关键修复：预测模式下，不要把当前目标行的 NaN 删掉
    if is_predict:
        # 只要求特征列不能有 NaN
        feature_cols = [c for c in df_feat.columns if c not in pollutants and not c.endswith('_delta')]
        return df_feat.dropna(subset=feature_cols).reset_index(drop=True), pollutants, meteo_cols
    else:
        # 训练模式下，要求所有列完整
        return df_feat.dropna().reset_index(drop=True), pollutants, meteo_cols

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1e-8
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100


# ---------------- 2. 空气质量 AQI 评级计算引擎 (贴合第三问要求) ----------------
def calculate_iaqi(C, pollutant):
    # 根据中国环境准则 HJ 633-2012 简化版的 IAQI 计算
    iaqi_bplist = [0, 50, 100, 150, 200, 300, 400, 500]
    bp_dict = {
        'PM2.5': [0, 35, 75, 115, 150, 250, 350, 500],
        'PM10': [0, 50, 150, 250, 350, 420, 500, 600],
        'O3': [0, 160, 200, 300, 400, 800, 1000, 1200],  # 1小时平均
        'NO2': [0, 100, 200, 700, 1200, 2340, 3090, 3840],
        'SO2': [0, 150, 500, 650, 800, 1600, 2100, 2620],
        'CO': [0, 5, 10, 35, 60, 90, 120, 150]  # mg/m3
    }
    bp = bp_dict.get(pollutant, bp_dict['PM2.5'])

    for i in range(1, len(bp)):
        if C <= bp[i]:
            # 线性插值公式
            iaqi = ((iaqi_bplist[i] - iaqi_bplist[i - 1]) / (bp[i] - bp[i - 1])) * (C - bp[i - 1]) + iaqi_bplist[i - 1]
            return np.ceil(iaqi)
    return 500  # 爆表值


def get_comprehensive_aqi(row):
    iaqis = [
        calculate_iaqi(row['PM2.5'], 'PM2.5'),
        calculate_iaqi(row['PM10'], 'PM10'),
        calculate_iaqi(row['O3'], 'O3'),
        calculate_iaqi(row['NO2'], 'NO2'),
        calculate_iaqi(row['SO2'], 'SO2'),
        calculate_iaqi(row['CO'], 'CO')  # 传入的是已经转回 mg/m3 的值
    ]
    return max(iaqis)


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


if __name__ == "__main__":
    print("开始读取数据 Adata2...")
    file_path = 'Adata2.xlsx'
    df = pd.read_excel(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 🌟 关键修复：强制将所有污染物列转为浮点数，防止向 int64 列插入 float 时报错
    pollutants_list = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    for p in pollutants_list:
        df[p] = pd.to_numeric(df[p], errors='coerce').astype('float64')

        # ---------------- 3. 模型训练 (仅使用 1-2 月数据) ----------------
        print("正在进行特征工程并训练基于 1-2 月的基座模型...")
        df_train_raw = df[df['datetime'].dt.month < 3].copy()
        df_train, pollutants, meteo_cols = build_features(df_train_raw)

        delta_cols = [f'{p}_delta' for p in pollutants]
        drop_cols = pollutants + delta_cols + ['datetime']

        # 特征矩阵 X 保持不变（不包含答案）
        X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])

        # 🌟 核心修改 1：目标矩阵 Y 改为直接预测绝对浓度！
        Y_train = df_train[pollutants]

        rf_base = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model = MultiOutputRegressor(rf_base)
        model.fit(X_train, Y_train)

    # ---------------- 4. 第三问：模拟生产环境的每日 22:00 滚动预测 ----------------
    print("\n" + "=" * 60)
    print("🔄 开始模拟三月份的真实生产环境滚动预测...")
    print("规则：每日 22:00 点，利用已知历史及明日天气，逐时迭代预测未来 24 小时")
    print("=" * 60)

    # 获取需要进行预测的锚点时间（每天22点）
    march_days = df[(df['datetime'].dt.month == 3)]['datetime'].dt.day.unique()

    # 储存整个三月的预测结果
    all_predictions = []

    for day in march_days:
        # 锚点时间: 前一天的 22:00 (如果day=1, 则是2月28日 22:00)
        if day == 1:
            anchor_time = pd.to_datetime('2026-02-28 22:00:00')
        else:
            anchor_time = pd.to_datetime(f'2026-03-{day - 1:02d} 22:00:00')

        # 预测窗口: 当晚 23:00 到次日 22:00 (共24小时)
        window_start = anchor_time + pd.Timedelta(hours=1)
        window_end = anchor_time + pd.Timedelta(hours=24)

        if window_end > df['datetime'].max():
            window_end = df['datetime'].max()  # 防止超出数据边界

        # 提取上下文数据 (过去48小时的真实历史 + 未来24小时的气象预报)
        # 注意：这里我们提取完整的 df，但在预测窗口内的污染物我们会将其抹去，模拟未知的未来
        context_start = anchor_time - pd.Timedelta(hours=48)
        df_work = df[(df['datetime'] >= context_start) & (df['datetime'] <= window_end)].copy()

        # 将未来 24 小时的污染物标记为 NaN (假装我们不知道)
        future_mask = df_work['datetime'] > anchor_time
        for p in pollutants:
            df_work.loc[future_mask, p] = np.nan

            # 开始自回归逐时迭代 (注意这里用了小写的 'h')
            # 开始自回归逐时迭代
            current_pred_times = pd.date_range(window_start, window_end, freq='h')
            for target_time in current_pred_times:
                # 取出截止到目标时间的数据
                df_temp = df_work[df_work['datetime'] <= target_time].copy()
                df_feat, _, _ = build_features(df_temp, is_predict=True)

                # 提取最后一行作为当前特征
                X_target = df_feat.iloc[[-1]].drop(columns=[c for c in drop_cols if c in df_feat.columns])

                # 🌟 防御性编程：强制特征列顺序与训练集对齐，防止 Pandas 打乱顺序导致预测错乱
                X_target = X_target[X_train.columns]

                # 🌟 核心修改 2：直接预测绝对值 (不再预测增量)
                y_pred = model.predict(X_target)[0]

                # 填入预测结果
                for i, p in enumerate(pollutants):
                    if p == 'CO':
                        # 特征工程时 CO 放大了 1000 倍，模型预测出的也是放大的，这里需要除以 1000 还原回 mg/m3
                        pred_val_true = max(0, y_pred[i] / 1000.0)
                    else:
                        pred_val_true = max(0, y_pred[i])  # 保证浓度不为负数

                    # 将预测出的当小时结果，填入工作台，供下一小时做滞后特征使用
                    df_work.loc[df_work['datetime'] == target_time, p] = pred_val_true

                # 记录该小时的预测结果
                row_result = {'datetime': target_time}
                for p in pollutants:
                    row_result[f'{p}_pred'] = df_work.loc[df_work['datetime'] == target_time, p].values[0]
                all_predictions.append(row_result)

        if day % 5 == 0 or day == 1:
            print(f"✅ 已完成预测截止至: {window_end.strftime('%Y-%m-%d %H:00')} 的数据")

    # ---------------- 5. 整理预测结果并计算三月评价指标 ----------------
    df_pred = pd.DataFrame(all_predictions)
    # 提取真实三月数据对齐
    df_true = df[df['datetime'].dt.month == 3][['datetime'] + pollutants].copy()

    # 合并预测与真实值
    df_eval = pd.merge(df_true, df_pred, on='datetime', how='inner')

    print("\n" + "=" * 80)
    print("🏆 三月全月预测准确率 (24小时滚动窗口验证)")
    print("=" * 80)
    print(f"{'污染物':<6} | {'模型 R2':<8} | {'RMSE':<8} | {'MAE':<8} | {'sMAPE(%)':<8}")
    print("-" * 80)

    eval_results = []
    for p in pollutants:
        y_true = df_eval[p].values
        y_pred = df_eval[f'{p}_pred'].values

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        s_mape = smape(y_true, y_pred)

        print(f"[{p:<5}] | {r2:<8.3f} | {rmse:<8.2f} | {mae:<8.2f} | {s_mape:<8.2f}")
        eval_results.append({'污染物': p, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'sMAPE(%)': s_mape})

    pd.DataFrame(eval_results).to_csv('Q3_March_Metrics.csv', index=False, encoding='utf-8-sig')

    # ---------------- 6. AQI 计算与安排活动建议 ----------------
    df_eval['True_AQI'] = df_eval.apply(get_comprehensive_aqi, axis=1)
    # 为了传给 get_comprehensive_aqi，重命名一下 pred 列作为临时列
    temp_pred_df = df_eval[['datetime'] + [f'{p}_pred' for p in pollutants]].rename(
        columns={f'{p}_pred': p for p in pollutants})
    df_eval['Pred_AQI'] = temp_pred_df.apply(get_comprehensive_aqi, axis=1)
    df_eval['活动建议'] = df_eval['Pred_AQI'].apply(get_aqi_level)

    # 输出总预测表
    df_eval.to_csv('Q3_March_Full_Predictions_with_AQI.csv', index=False, encoding='utf-8-sig')
    print("✅ 三月全月预测总表（含AQI及活动建议）已保存至 'Q3_March_Full_Predictions_with_AQI.csv'")

    # ---------------- 7. 最后五天高大上图表绘制 ----------------
    print("\n📊 正在绘制最后 5 天 (3.27 - 3.31) 综合趋势对比图...")
    df_plot = df_eval[df_eval['datetime'] >= '2026-03-27 00:00:00'].set_index('datetime')

    # 图 1：综合 AQI 对比图
    plt.figure(figsize=(14, 5))
    plt.plot(df_plot.index, df_plot['True_AQI'], label='真实 AQI (True)', color='black', linewidth=2, marker='o',
             markersize=4)
    plt.plot(df_plot.index, df_plot['Pred_AQI'], label='预测 AQI (Predicted)', color='#FF5722', linewidth=2,
             linestyle='--', marker='^', markersize=4)
    plt.axhline(y=100, color='red', linestyle=':', label='污染警戒线 (AQI=100)')
    plt.fill_between(df_plot.index, df_plot['True_AQI'], df_plot['Pred_AQI'], color='gray', alpha=0.1)
    plt.title('良乡校区最后5天综合空气质量指数 (AQI) 滚动预测对比', fontsize=16)
    plt.ylabel('AQI 指数')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Q3_Last5Days_AQI_Trend.png', dpi=300)

    # 图 2：六项污染物独立趋势矩阵
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, p in enumerate(pollutants):
        ax = axes[i]
        ax.plot(df_plot.index, df_plot[p], label='真实值', color='black', linewidth=1.5)
        ax.plot(df_plot.index, df_plot[f'{p}_pred'], label='预测值', color=colors[i], linewidth=2, linestyle='--')

        unit = 'mg/m³' if p == 'CO' else 'μg/m³'
        ax.set_title(f'{p} 浓度变化趋势', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'浓度 ({unit})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('最后5天六项污染物协同滚动预测对比 (24小时动态刷新)', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig('Q3_Last5Days_Pollutants_Matrix.png', dpi=300)

    print("✅ 绘图完成！图表已保存为 'Q3_Last5Days_AQI_Trend.png' 和 'Q3_Last5Days_Pollutants_Matrix.png'")