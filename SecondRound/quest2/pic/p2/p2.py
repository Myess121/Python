import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# 全局字体与格式设置
# =========================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei', font_scale=1.1)

print("📂 正在读取消融实验对比数据...")
try:
    # 读取原版（含 Y_t）和 盲测版（无 Y_t）的数据
    df_orig = pd.read_csv('RF_Model_Results_V2.csv')
    df_new = pd.read_csv('RF_no_yt_results.csv')
except Exception as e:
    print(f"读取文件出错: {e}")
    exit()

pollutants = df_orig['污染物'].values
# 获取两者的 R2 分数
r2_orig = df_orig['模型 R2'].values
r2_new = df_new['模型 R2'].values

x = np.arange(len(pollutants))
width = 0.35

# =========================================
# 开始绘图：特征消融稳定性对比图
# =========================================
print("正在生成特征消融稳定性对比图...")
fig, ax = plt.subplots(figsize=(10, 6.5))

# 原版模型 (深绿色)
rects1 = ax.bar(x - width/2, r2_orig, width, label='原版模型 (包含历史浓度 $Y_t$)', color='#55A868', edgecolor='black', linewidth=0.8)
# 消融模型 (浅绿色，代表剥离了部分特征，但依然坚挺)
rects2 = ax.bar(x + width/2, r2_new, width, label='消融模型 (剔除 $Y_t$，纯气象驱动)', color='#A8D08D', edgecolor='black', linewidth=0.8)

ax.set_ylabel('决定系数 ($R^2$)', fontsize=13, fontweight='bold')
ax.set_title('特征消融实验：剔除自回归项后的模型稳定性检验', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(pollutants, fontsize=12, fontweight='bold')

# 动态调整 Y 轴下限，让“微小的差距”能看清楚，同时包容 SO2 的 0.84
ax.set_ylim(0.75, 1.1)

# 统一沉底图例
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=True, shadow=True, fontsize=12)

# 标注数值
for rect in rects1:
    ax.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.005, f'{rect.get_height():.3f}', ha='center', va='bottom', fontsize=10, color='#2F5F3A', fontweight='bold')
for rect in rects2:
    ax.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.005, f'{rect.get_height():.3f}', ha='center', va='bottom', fontsize=10, color='#507A3E')

plt.savefig('Fig6_Ablation_Study_R2.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ 大功告成！消融实验对比图 (Fig6_Ablation_Study_R2.png) 已保存！")