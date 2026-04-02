import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import Slider, Button
from matplotlib.ticker import ScalarFormatter

# ================= 1. 环境设置 =================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 全局状态管理 =================
GRID_SIZE = 60
LOOKAHEAD_RADIUS = 18  # 搜索视野
MAX_UAVS = 5
COLORS = ['cyan', 'lime', 'magenta', 'yellow', 'white']

state = {}


def init_state():
    # 初始概率分布：均匀分布
    state['prob_map'] = np.ones((GRID_SIZE, GRID_SIZE)) / (GRID_SIZE ** 2)
    state['uav_pos'] = np.zeros((MAX_UAVS, 2))
    state['trajectories'] = [[] for _ in range(MAX_UAVS)]

    for i in range(MAX_UAVS):
        state['uav_pos'][i] = [2.0 + i * 3, 2.0]
        state['trajectories'][i] = [state['uav_pos'][i].copy()]

    state['step'] = 0
    state['flight_time'] = 0.0
    state['total_dist'] = 0.0
    # 用于检测失败的滑动窗口
    state['prob_history'] = []

    state['running'] = False
    state['started'] = False
    state['reset_flag'] = False


init_state()

# ================= 3. 创建 UI 界面 =================
fig, ax = plt.subplots(figsize=(11, 8.5))
plt.subplots_adjust(bottom=0.4, right=0.82)

# 固定 vmax=0.0005 确保颜色条刻度不随程序运行而变动
im = ax.imshow(state['prob_map'], cmap='inferno', origin='lower', vmin=0, vmax=0.0005)

# 固定右侧颜色条
cbar_ax = fig.add_axes([0.85, 0.45, 0.02, 0.4])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('目标存在概率密度', fontsize=12)

# 强制消除科学计数法（防止出现 1e-5）
formatter = ScalarFormatter(useMathText=False)
formatter.set_scientific(False)
cbar.ax.yaxis.set_major_formatter(formatter)

uav_scatters = []
sensor_circles = []
traj_lines = []

for i in range(MAX_UAVS):
    line, = ax.plot([], [], color=COLORS[i], lw=1.2, alpha=0.6, label=f'无人机 {i + 1}' if i == 0 else "")
    traj_lines.append(line)
    scatter, = ax.plot([], [], 'o', color=COLORS[i], markersize=5)
    uav_scatters.append(scatter)
    circle = plt.Circle((0, 0), 0, color=COLORS[i], fill=False, lw=1, alpha=0.5)
    ax.add_patch(circle)
    sensor_circles.append(circle)

# 中央提示文字
alert_text = ax.text(GRID_SIZE / 2, GRID_SIZE / 2, '', color='white', fontsize=18,
                     ha='center', va='center', weight='bold', zorder=15)
alert_text.set_visible(False)

ax.set_title("区域概率覆盖仿真 - 待机中")

# ================= 4. 控制面板 =================
ax_num = plt.axes([0.15, 0.28, 0.65, 0.03], facecolor='lightgray')
ax_speed = plt.axes([0.15, 0.23, 0.65, 0.03], facecolor='lightgray')
ax_radius = plt.axes([0.15, 0.18, 0.65, 0.03], facecolor='lightgray')
ax_target = plt.axes([0.15, 0.13, 0.65, 0.03], facecolor='lightgray')

sl_num = Slider(ax_num, '飞机数量', 1, MAX_UAVS, valinit=3, valstep=1)
sl_speed = Slider(ax_speed, '飞行速度', 0.5, 5.0, valinit=2.5)
sl_radius = Slider(ax_radius, '雷达半径', 2.0, 10.0, valinit=5.5)
sl_target = Slider(ax_target, '目标扩散速度', 0.1, 1.5, valinit=0.7)

ax_start = plt.axes([0.2, 0.04, 0.15, 0.05])
ax_pause = plt.axes([0.45, 0.04, 0.15, 0.05])
ax_reset = plt.axes([0.7, 0.04, 0.15, 0.05])

btn_start = Button(ax_start, '▶ 开始搜索', color='palegreen')
btn_pause = Button(ax_pause, '⏸ 暂停/继续', color='skyblue')
btn_reset = Button(ax_reset, '🔄 重置地图', color='salmon')


def cb_start(event):
    state['started'] = True
    state['running'] = True
    alert_text.set_visible(False)


def cb_pause(event):
    if state['started']:
        state['running'] = not state['running']


def cb_reset(event):
    state['reset_flag'] = True


btn_start.on_clicked(cb_start)
btn_pause.on_clicked(cb_pause)
btn_reset.on_clicked(cb_reset)

# ================= 5. 核心逻辑循环 =================
plt.ion()

while plt.fignum_exists(fig.number):
    if state['reset_flag']:
        init_state()
        alert_text.set_visible(False)
        im.set_data(state['prob_map'])
        for i in range(MAX_UAVS):
            traj_lines[i].set_data([], [])
            uav_scatters[i].set_data([], [])
            sensor_circles[i].set_radius(0)
        ax.set_title("地图已重置，请调整参数后点击【开始】")
        fig.canvas.draw_idle()
        continue

    if not state['running']:
        plt.pause(0.1)
        continue

    try:
        cur_num = int(sl_num.val)
        cur_speed = sl_speed.val
        cur_radius = sl_radius.val
        cur_diffuse = sl_target.val

        # 1. 概率马尔可夫扩散（代表目标可能的移动导致不确定性增加）
        state['prob_map'] = gaussian_filter(state['prob_map'], sigma=cur_diffuse)

        # 2. 无人机智能寻优 (RHC + 协同排斥)
        decision_map = np.copy(state['prob_map'])
        Y, X = np.ogrid[:GRID_SIZE, :GRID_SIZE]

        for i in range(cur_num):
            # 局部视野内找最高点
            dist_sq = (Y - state['uav_pos'][i][0]) ** 2 + (X - state['uav_pos'][i][1]) ** 2
            local_mask = dist_sq <= LOOKAHEAD_RADIUS ** 2

            if np.sum(decision_map[local_mask]) > 1e-7:
                local_prob = np.copy(decision_map)
                local_prob[~local_mask] = 0
                target_idx = np.unravel_index(np.argmax(local_prob), local_prob.shape)
            else:
                # 视野内全黑则寻找全局最高点
                target_idx = np.unravel_index(np.argmax(decision_map), decision_map.shape)

            target_pos = np.array(target_idx)

            # 【优化】协同逻辑：一架飞机选定目标后，暂时削弱该区对队友的吸引力
            repel_mask = ((Y - target_pos[0]) ** 2 + (X - target_pos[1]) ** 2) <= (cur_radius * 2) ** 2
            decision_map[repel_mask] *= 0.1

            # 物理机动
            move_vec = target_pos - state['uav_pos'][i]
            d = np.linalg.norm(move_vec)
            if d > 0:
                step_move = (move_vec / d) * min(cur_speed, d)
                state['uav_pos'][i] += step_move
                state['total_dist'] += np.linalg.norm(step_move)

            state['trajectories'][i].append(state['uav_pos'][i].copy())

            # 贝叶斯探测：清空雷达覆盖区的概率
            sensor_mask = ((Y - state['uav_pos'][i][0]) ** 2 + (X - state['uav_pos'][i][1]) ** 2) <= cur_radius ** 2
            state['prob_map'][sensor_mask] = 0

        # 3. 统计指标与终止判定
        remaining_prob = np.sum(state['prob_map'])
        state['prob_history'].append(remaining_prob)
        if len(state['prob_history']) > 100:  # 考察最近100步的趋势
            state['prob_history'].pop(0)

        state['flight_time'] += 1.0

        # --- 判定逻辑 ---
        # 成功：剩余概率极低 (低于 0.1%)
        if remaining_prob < 0.001:
            alert_text.set_text("【寻找成功】\n全域搜索进度已达 100%")
            alert_text.set_bbox(dict(facecolor='green', alpha=0.8, edgecolor='white'))
            alert_text.set_visible(True)
            state['running'] = False

        # 失败：运行了一段时间，且概率不再下降（进入稳态）
        elif state['step'] > 120 and (max(state['prob_history']) - min(state['prob_history']) < 0.0005):
            alert_text.set_text("【寻找失败】\n剩余概率已稳定，无法继续降低")
            alert_text.set_bbox(dict(facecolor='red', alpha=0.8, edgecolor='white'))
            alert_text.set_visible(True)
            state['running'] = False

        # 4. 绘图刷新
        if state['step'] % 2 == 0:
            im.set_data(state['prob_map'])
            for i in range(MAX_UAVS):
                if i < cur_num:
                    ty, tx = zip(*state['trajectories'][i])
                    traj_lines[i].set_data(tx, ty)
                    uav_scatters[i].set_data([state['uav_pos'][i][1]], [state['uav_pos'][i][0]])
                    sensor_circles[i].set_center((state['uav_pos'][i][1], state['uav_pos'][i][0]))
                    sensor_circles[i].set_radius(cur_radius)
                    traj_lines[i].set_visible(True);
                    uav_scatters[i].set_visible(True);
                    sensor_circles[i].set_visible(True)
                else:
                    traj_lines[i].set_visible(False);
                    uav_scatters[i].set_visible(False);
                    sensor_circles[i].set_visible(False)

            ax.set_title(
                f"时间: {state['flight_time']:.0f}min | 总路程: {state['total_dist']:.0f}km | 剩余不确定性: {remaining_prob:.2%}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        state['step'] += 1
        plt.pause(0.01)

    except Exception as e:
        print(f"Error: {e}")
        break

plt.ioff()
plt.show()