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


def build_features(df):
    df_feat = df.copy()

    # ------------------ 新增的脏数据清洗补丁 ------------------
    # 1. 强制转换为字符串格式
    df_feat['datetime'] = df_feat['datetime'].astype(str)
    # 2. 暴力修复官方数据里的录入错误（把多余的空格删掉）
    df_feat['datetime'] = df_feat['datetime'].str.replace('2 026', '2026')
    # --------------------------------------------------------
    # 1. 先确保是字符串，处理可能存在的空格
    df_feat['datetime'] = df_feat['datetime'].astype(str).str.strip()

    # 2. 修复特定的录入错误
    df_feat['datetime'] = df_feat['datetime'].str.replace('2 026', '2026')

    # 3. 使用 mixed 模式转换，并对无法转换的错误值填入 NaT (errors='coerce')
    df_feat['datetime'] = pd.to_datetime(df_feat['datetime'], format='mixed', errors='coerce')

    # 4. 删除由于时间格式彻底错误产生的空行
    df_feat = df_feat.dropna(subset=['datetime'])


    temp_lags = [1, 6, 12, 13, 14]
    for lag in temp_lags:
        df_feat[f'Temp_lag{lag}'] = df_feat['temperature'].shift(lag)

    hum_lags = [1, 13, 24]
    for lag in hum_lags:
        df_feat[f'Hum_lag{lag}'] = df_feat['humidity'].shift(lag)

    pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']

    # 【补丁 2】：统一 CO 量纲 (mg -> ug) 参与训练
    df_feat['CO'] = pd.to_numeric(df_feat['CO'], errors='coerce') * 1000

    for p in pollutants:
        if p != 'CO':
            df_feat[p] = pd.to_numeric(df_feat[p], errors='coerce')

        # 构造滞后 1 小时特征 (用于作为基准还原预测值)
        df_feat[f'{p}_lag1'] = df_feat[p].shift(1)

        # 【补丁 1】：构造目标变量 ΔY (浓度变化量)
        df_feat[f'{p}_delta'] = df_feat[p] - df_feat[f'{p}_lag1']

    return df_feat.dropna(), pollutants


def calculate_smape(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8)) * 100


if __name__ == "__main__":
    print("开始读取数据...")
    file_path = '4-2026年校赛第二轮A题数据.xlsx'
    df = pd.read_excel(file_path, skiprows=9)

    print("正在进行特征工程 (包含 ΔY 差分转换)...")
    df_processed, pollutants = build_features(df)

    # X：特征矩阵 (不包含当期真实浓度 和 当期变化量)
    delta_cols = [f'{p}_delta' for p in pollutants]
    drop_cols = pollutants + delta_cols + ['datetime', 'hour', 'temperature', 'humidity']
    drop_cols = [c for c in drop_cols if c in df_processed.columns]
    X = df_processed.drop(columns=drop_cols)

    # Y：目标矩阵 (这次预测的是变化量 ΔY)
    Y_delta = df_processed[delta_cols]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_delta_train, Y_delta_test = Y_delta.iloc[:split_idx], Y_delta.iloc[split_idx:]

    print("正在训练基于差分的协同随机森林模型...")
    rf_base = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(rf_base)
    model.fit(X_train, Y_delta_train)

    print("预测完毕，正在还原真实浓度与真实量纲...")
    Y_delta_pred = model.predict(X_test)
    Y_delta_pred_df = pd.DataFrame(Y_delta_pred, columns=delta_cols, index=X_test.index)

    # 【必须修复的硬伤】：使用 .loc 严格按索引对齐，防止错位
    Y_pred_real = pd.DataFrame(index=X_test.index)
    Y_test_real = df_processed[pollutants].loc[X_test.index]

    for p in pollutants:
        Y_pred_real[p] = X_test[f'{p}_lag1'] + Y_delta_pred_df[f'{p}_delta']

    Y_pred_real['CO'] = Y_pred_real['CO'] / 1000
    Y_test_real['CO'] = Y_test_real['CO'] / 1000

    # ---------------------------------------------------------
    # 【高光时刻】：模型竞技场 (Baseline vs 本文模型)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("🏆 模型竞技场对比 (持久性基准 Baseline VS 本文气象驱动模型)")
    print("=" * 80)

    metrics_dict = {}  # 用于保存指标方便后续画图或存表

    for p in pollutants:
        # 计算本文模型的指标
        r2_model = r2_score(Y_test_real[p], Y_pred_real[p])
        rmse_model = np.sqrt(mean_squared_error(Y_test_real[p], Y_pred_real[p]))

        # 计算“最蠢”基准的指标 (即预测下一时刻等于上一时刻)
        # 注意：因为 CO 在前面统一乘以了 1000，这里的基准值也要记得除以 1000 还原
        if p == 'CO':
            y_base = X_test[f'{p}_lag1'].values / 1000
        else:
            y_base = X_test[f'{p}_lag1'].values

        r2_base = r2_score(Y_test_real[p], y_base)
        rmse_base = np.sqrt(mean_squared_error(Y_test_real[p], y_base))

        # 计算提升率 (误差降低了多少)
        rmse_imp = (rmse_base - rmse_model) / rmse_base * 100

        print(
            f"[{p:<5}] | 本文 R2: {r2_model:.3f} (基准: {r2_base:.3f}) | 本文 RMSE: {rmse_model:<6.2f} (基准: {rmse_base:<6.2f}) -> 误差降低: {rmse_imp:.1f}%")

        metrics_dict[p] = {'R2_model': r2_model, 'R2_base': r2_base, 'RMSE_imp_percent': rmse_imp}

    print("=" * 80)

    # 计算 Frobenius 范数
    corr_true = Y_test_real.corr().values
    corr_pred = Y_pred_real.corr().values
    frob_norm = np.linalg.norm(corr_true - corr_pred)
    print(f"\n【协同矩阵指标】真实与预测相关性矩阵的 Frobenius 范数距离: {frob_norm:.4f}")

    # ... 后面的画图代码保持不变 ...



    print("\n🔄 动态协同验证 (预测变化量 ΔY 的相关性):")
    delta_corr_true = Y_delta_test.corr().values
    delta_corr_pred = Y_delta_pred_df.corr().values
    frob_delta = np.linalg.norm(delta_corr_true - delta_corr_pred)
    print(f"ΔY 协同矩阵 Frobenius 距离: {frob_delta:.4f} (证明模型捕获的是'变化驱动'而非'静态惯性')")

    # 出图：ΔY 协同对比
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(delta_corr_true, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes2[0], fmt='.2f', xticklabels=pollutants, yticklabels=pollutants)
    axes2[0].set_title("真实浓度变化量(ΔY)协同矩阵")
    sns.heatmap(delta_corr_pred, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes2[1], fmt='.2f', xticklabels=pollutants, yticklabels=pollutants)
    axes2[1].set_title("模型预测变化量(ΔY)协同矩阵")
    plt.tight_layout()
    plt.savefig('Q2_Synergy_Delta_Correlation.png', dpi=300, bbox_inches='tight')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(Y_test_real.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0], fmt='.2f')
    axes[0].set_title("真实浓度相关性矩阵")

    sns.heatmap(Y_pred_real.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1], fmt='.2f')
    axes[1].set_title("模型预测相关性矩阵")

    plt.tight_layout()
    plt.savefig('Q2_Synergy_Correlation.png', dpi=300)
    print("\n图片已保存为 'Q2_Synergy_Correlation.png'")

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