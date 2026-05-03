import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# 全局字体与格式设置
# =========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False     # 负号显示
sns.set_theme(style="whitegrid", font='SimHei', font_scale=1.1)

# =========================================
# 智能读取数据
# =========================================
print("📂 正在读取数据...")
try:
    df_p1 = pd.read_csv('Phase1_Pure_Meteo_Results.csv')
    df_rf = pd.read_csv('RF_Model_Results_V2.csv')
    df_lstm = pd.read_csv('LSTM_Comparison_Results_Final.csv')
except Exception as e:
    print(f"读取文件出错，请确保这三个 CSV 都在当前文件夹里: {e}")
    exit()

pollutants = df_p1['污染物'].values
r2_meteo = df_p1['纯气象 R2'].values
r2_inertia = df_p1['基准(惯性) R2'].values

# 提取 RF 和 LSTM 核心指标
col_rf_r2 = [c for c in df_rf.columns if 'R2' in c and '基准' not in c][0]
col_lstm_r2 = [c for c in df_lstm.columns if 'R2' in c and '基准' not in c][0]
col_rf_imp = [c for c in df_rf.columns if '误差降低' in c][0]
col_lstm_imp = [c for c in df_lstm.columns if '误差降低' in c][0]
col_rf_smape = [c for c in df_rf.columns if 'sMAPE' in c][0]

r2_rf = df_rf[col_rf_r2].values
r2_lstm = df_lstm[col_lstm_r2].values
imp_rf = df_rf[col_rf_imp].values
imp_lstm = df_lstm[col_lstm_imp].values
smape_rf = df_rf[col_rf_smape].values

x = np.arange(len(pollutants))

# =========================================
# 方案 1：无遮挡版·三阶段演进柱状图
# =========================================
print("正在生成图1...")
fig1, ax1 = plt.subplots(figsize=(12, 7)) # 画布放大
width = 0.25

rects1 = ax1.bar(x - width, r2_meteo, width, label='阶段一：纯气象驱动', color='#4C72B0', edgecolor='black', linewidth=0.8)
rects2 = ax1.bar(x, r2_inertia, width, label='阶段二：纯惯性基准', color='#DD8452', edgecolor='black', linewidth=0.8)
rects3 = ax1.bar(x + width, r2_rf, width, label='阶段三：气象联合差分架构 (RF)', color='#55A868', edgecolor='black', linewidth=0.8)

ax1.set_ylabel('决定系数 ($R^2$)', fontsize=13, fontweight='bold')
ax1.set_title('多污染物预测模型：驱动力三阶段演进对比分析', fontsize=16, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(pollutants, fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.15)

# 🌟 统一沉底：放到 X 轴下方 (y = -0.12)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=True, shadow=True, fontsize=12)

for rect in rects1: ax1.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.01, f'{rect.get_height():.2f}', ha='center', va='bottom', fontsize=10)
for rect in rects2: ax1.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.01, f'{rect.get_height():.2f}', ha='center', va='bottom', fontsize=10)
for rect in rects3: ax1.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.01, f'{rect.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.savefig('Option1_Three_Phases_Bar_BottomLeg.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================================
# 方案 2：哑铃图 (Dumbbell Plot)
# =========================================
print("正在生成图2...")
fig2, ax2 = plt.subplots(figsize=(10, 6.5))

for i in range(len(pollutants)):
    ax2.plot([imp_lstm[i], imp_rf[i]], [i, i], color='grey', zorder=1, linestyle='--', alpha=0.7)

ax2.scatter(imp_lstm, x, color='#8172B3', s=150, zorder=2, label='对照模型 (LSTM-ΔY)', edgecolor='black')
ax2.scatter(imp_rf, x, color='#C44E52', s=150, zorder=2, label='本文模型 (RF-ΔY)', edgecolor='black')

ax2.set_yticks(x)
ax2.set_yticklabels(pollutants, fontsize=12, fontweight='bold')
ax2.set_xlabel('RMSE 误差降低率 (%)', fontsize=13, fontweight='bold')
ax2.set_title('控制变量下深度学习与集成学习性能差异评估', fontsize=16, fontweight='bold', pad=15)

# 🌟 统一沉底
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True, shadow=True, fontsize=12)

for i in range(len(pollutants)):
    ax2.text(imp_lstm[i], i + 0.25, f'{imp_lstm[i]:.1f}%', ha='center', va='center', color='#8172B3', fontsize=10)
    ax2.text(imp_rf[i], i + 0.25, f'{imp_rf[i]:.1f}%', ha='center', va='center', color='#C44E52', fontweight='bold', fontsize=10)

plt.savefig('Option2_Dumbbell_Gap_BottomLeg.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================================
# 方案 3：水平条形图
# =========================================
print("正在生成图3...")
fig3, ax3 = plt.subplots(figsize=(10, 7.5))
bar_height = 0.35

rects_rf = ax3.barh(x + bar_height/2, imp_rf, bar_height, label='本文模型 (RF-ΔY)', color='#C44E52', edgecolor='black')
rects_lstm = ax3.barh(x - bar_height/2, imp_lstm, bar_height, label='对照模型 (LSTM-ΔY)', color='#8172B3', edgecolor='black')

ax3.set_xlabel('RMSE 误差降低率 (%)', fontsize=13, fontweight='bold')
ax3.set_title('多源气象特征驱动下的模型误差降低率评估', fontsize=16, fontweight='bold', pad=15)
ax3.set_yticks(x)
ax3.set_yticklabels(pollutants, fontsize=12, fontweight='bold')
ax3.set_xlim(0, max(imp_rf.max(), imp_lstm.max()) * 1.2)

# 🌟 统一沉底
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, shadow=True, fontsize=12)

for rect in rects_rf: ax3.text(rect.get_width() + 1.5, rect.get_y() + rect.get_height()/2., f'{rect.get_width():.1f}%', va='center', fontweight='bold', color='#C44E52')
for rect in rects_lstm: ax3.text(rect.get_width() + 1.5, rect.get_y() + rect.get_height()/2., f'{rect.get_width():.1f}%', va='center', color='#8172B3')

plt.savefig('Option3_Horizontal_Bars_BottomLeg.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================================
# 方案 4：双轴综合图 (Dual Axis)
# =========================================
print("正在生成图4...")
fig4, ax_left = plt.subplots(figsize=(12, 7)) # 画布拉宽拉高
ax_right = ax_left.twinx()

rects_r2 = ax_left.bar(x, r2_rf, width=0.4, color='#55A868', alpha=0.8, edgecolor='black', label='本文模型 $R^2$ (左轴)')
ax_left.set_ylabel('决定系数 ($R^2$)', fontsize=13, fontweight='bold', color='#2F5F3A')
ax_left.set_ylim(0, 1.15)

line_smape = ax_right.plot(x, smape_rf, color='#C44E52', marker='o', markersize=8, linewidth=2.5, label='对称绝对百分比误差 sMAPE (右轴)')
ax_right.set_ylabel('sMAPE (%)', fontsize=13, fontweight='bold', color='#C44E52')
ax_right.set_ylim(0, max(smape_rf)*1.3)

ax_left.set_xticks(x)
ax_left.set_xticklabels(pollutants, fontsize=12, fontweight='bold')
ax_left.set_title('本文最终模型 (RF-ΔY) 多维综合性能评估', fontsize=16, fontweight='bold', pad=15)

# 🌟 统一沉底：合并双轴的图例，一起扔到图表下方
lines, labels = ax_left.get_legend_handles_labels()
lines2, labels2 = ax_right.get_legend_handles_labels()
ax_left.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=True, shadow=True, fontsize=12)

for rect in rects_r2: ax_left.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.02, f'{rect.get_height():.3f}', ha='center', color='#2F5F3A')
for i, txt in enumerate(smape_rf): ax_right.text(x[i], smape_rf[i] + 0.6, f'{txt:.1f}%', ha='center', fontweight='bold', color='#C44E52')

plt.savefig('Option4_Dual_Axis_BottomLeg.png', dpi=300, bbox_inches='tight') # bbox_inches='tight' 保证底下图例不被切掉
plt.close()

# =========================================
# 方案 5：多维学术雷达图 (Radar Chart)
# =========================================
print("正在生成图5...")
angles = np.linspace(0, 2 * np.pi, len(pollutants), endpoint=False).tolist()
angles += angles[:1]
r2_rf_circ = np.append(r2_rf, r2_rf[0])
r2_lstm_circ = np.append(r2_lstm, r2_lstm[0])
r2_base_circ = np.append(r2_inertia, r2_inertia[0])

fig5, ax5 = plt.subplots(figsize=(9, 9))
ax5 = plt.subplot(111, polar=True)
ax5.plot(angles, r2_rf_circ, color='#55A868', linewidth=2.5, label='本文模型 (RF-ΔY)')
ax5.fill(angles, r2_rf_circ, color='#55A868', alpha=0.2)
ax5.plot(angles, r2_lstm_circ, color='#8172B3', linewidth=2, linestyle='--', label='对照模型 (LSTM-ΔY)')
ax5.plot(angles, r2_base_circ, color='#DD8452', linewidth=2, linestyle=':', label='基准模型 (纯惯性)')

ax5.set_yticklabels([])
ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(pollutants, fontsize=13, fontweight='bold')
ax5.set_title("各模型对污染物的泛化覆盖能力评估雷达图", fontsize=16, fontweight='bold', pad=25)

# 🌟 统一沉底
ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, shadow=True, fontsize=11)

plt.savefig('Option5_Radar_Plot_BottomLeg.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ 大功告成！所有图例全部沉底，防遮挡最强护甲已实装！")