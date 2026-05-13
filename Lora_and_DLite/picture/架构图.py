import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ================= 1. 画布与基本设置 =================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.set_xlim(0, 50)
ax.set_ylim(0, 30)

# 开启栅格
ax.grid(True, linestyle='--', color='gray', alpha=0.3)
ax.set_xticks(np.arange(0, 51, 5))
ax.set_yticks(np.arange(0, 31, 5))
ax.set_xlabel('X Coordinate (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y Coordinate (m)', fontsize=12, fontweight='bold')
# 增加 pad 参数，防止标题被顶部的框线切到
ax.set_title('System Architecture: Comm-Aware Path Planning', fontsize=14, fontweight='bold', pad=15)

# ================= 2. 添加环境实体 =================
obstacles = [
    patches.Rectangle((15, 10), 5, 8, facecolor='dimgray', edgecolor='black', alpha=0.8),
    patches.Rectangle((30, 5), 6, 6, facecolor='dimgray', edgecolor='black', alpha=0.8),
    patches.Rectangle((35, 18), 5, 7, facecolor='dimgray', edgecolor='black', alpha=0.8)
]
for obs in obstacles:
    ax.add_patch(obs)
ax.scatter([], [], c='dimgray', marker='s', s=100, label='Static Obstacles ($\mathcal{O}$)')

# 2.2 干扰源 (Jammer) 与干扰场
jammer_pos = (25, 15)
interference_zone = patches.Circle(jammer_pos, radius=8, color='red', alpha=0.2, label='Low LQM Zone (Interference)')
ax.add_patch(interference_zone)
ax.scatter(*jammer_pos, c='red', marker='X', s=150, zorder=5, label='Dynamic Jammer ($J$)')
ax.arrow(25, 15, 2, 3, head_width=1, head_length=1.5, fc='red', ec='red', zorder=5)

# 2.3 地面站 (Ground Station)
gs_pos = (5, 5)
ax.scatter(*gs_pos, c='blue', marker='^', s=200, zorder=5, label='Ground Station (GS)')

# ================= 3. 添加路径与无人机 =================
start_pos = (5, 25)
goal_pos = (45, 10)

ax.scatter(*start_pos, c='black', marker='o', s=80, zorder=5)
ax.text(start_pos[0]-1, start_pos[1]+1.5, 'Start ($s$)', fontsize=12, fontweight='bold')
ax.scatter(*goal_pos, c='black', marker='*', s=150, zorder=5)
ax.text(goal_pos[0]+1, goal_pos[1]+1, 'Goal ($g$)', fontsize=12, fontweight='bold')

# 3.1 传统最短路径
ax.plot([start_pos[0], goal_pos[0]], [start_pos[1], goal_pos[1]],
        linestyle='--', color='gray', linewidth=2.5, label='Standard D* Lite (Comm Break)')
ax.scatter(25, 17.5, c='red', marker='x', s=200, linewidths=3, zorder=6)

# 3.2 Comm-Aware D* Lite 路径
curved_path_x = [5, 12, 25, 33, 45]
curved_path_y = [25, 26, 26.5, 18, 10]
ax.plot(curved_path_x, curved_path_y, color='forestgreen', linewidth=3, zorder=4, label='Comm-Aware D* Lite (Safe)')

# 3.3 无人机当前位置
uav_pos = (25, 26.5)
ax.scatter(*uav_pos, c='orange', marker='h', s=180, edgecolor='black', zorder=6, label='UAV Swarm')

# ================= 4. 通信链路示意 =================
ax.plot([gs_pos[0], uav_pos[0]], [gs_pos[1], uav_pos[1]],
        linestyle=':', color='blue', linewidth=2, zorder=3)
ax.text(12, 16, 'LoRa Link\n(LQM > Threshold)', color='blue', fontsize=10, fontweight='bold', rotation=45)

# ================= 5. 图例与输出 =================
# 【核心修改点】将图例移到图表外部正下方，并分为3列
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3,
          fontsize=10, framealpha=0.9, edgecolor='black')

# 调整子图参数，给底部的图例留出空间
plt.subplots_adjust(bottom=0.25)

plt.savefig('System_Architecture.png', dpi=300, bbox_inches='tight')
plt.savefig('System_Architecture.pdf', bbox_inches='tight')

print("图片生成成功！已修复遮挡问题。")
plt.show()