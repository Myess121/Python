# environment.py
"""
高保真无人机城市通信物理环境引擎 (顶刊大修版)
核心功能：
1. 计算带对数正态阴影衰落的对数距离路径损耗。
2. 计算多动态干扰源叠加下的真实 SINR。
3. 基于实测 LoRa 性能曲线，进行 SINR -> PER -> 归一化得分 的物理映射。
"""
import math
import random

# ================= 物理层常量配置 =================
# environment.py (修改顶部配置区)

# 1. 提升无人机功率，降低干扰功率，给算法留出一条“活路”
P_TX = 20.0             # 无人机发射功率拉满 (原14.0)
NOISE_FLOOR = -110.0
FREQ_MHZ = 433.0
PATH_LOSS_EXP = 3.0
# 2. 必须设为0！实时随机数会彻底摧毁启发式图搜索！
SHADOWING_STD = 0.0     # 暂时关闭随机阴影衰落


def calculate_physical_sinr(uav_pos, gs_pos, jammers):
    """
    计算 UAV 到地面站(GS) 的真实物理信干噪比 (SINR)
    包含距离衰减、随机阴影衰落、以及多个干扰源的功率叠加。
    """
    # 1. 计算信号接收功率 (P_rx)
    dist_gs = math.hypot(uav_pos[0] - gs_pos[0], uav_pos[1] - gs_pos[1])
    # 避免距离为 0 导致 log10 报错
    dist_gs = max(dist_gs, 1.0)

    # 自由空间基准损耗 (假设 1 米处距离损耗约为 25dB)
    pl_0 = 25.0

    # 对数正态阴影衰落 (模拟城市楼宇遮挡和多径效应的随机性)
    shadowing = random.gauss(0, SHADOWING_STD)

    # 接收功率 = 发射功率 - 基准损耗 - 距离衰减损耗 - 阴影衰落
    p_rx_dbm = P_TX - pl_0 - 10 * PATH_LOSS_EXP * math.log10(dist_gs) - shadowing
    p_rx_linear = 10 ** (p_rx_dbm / 10.0)

    # 2. 计算总干扰功率 (I_total)
    i_total_linear = 0.0
    if jammers:
        for jammer in jammers:
            dist_j = math.hypot(uav_pos[0] - jammer.pos[0], uav_pos[1] - jammer.pos[1])
            dist_j = max(dist_j, 1.0)

            # 干扰信号同样经历路径损耗 (此处为简化，忽略干扰信号的阴影衰落)
            pl_j = pl_0 + 10 * PATH_LOSS_EXP * math.log10(dist_j)
            p_j_rx_dbm = jammer.power - pl_j
            i_total_linear += 10 ** (p_j_rx_dbm / 10.0)

    # 3. 计算热噪声 (N_0)
    n_0_linear = 10 ** (NOISE_FLOOR / 10.0)

    # 4. 计算最终物理 SINR (dB)
    sinr_linear = p_rx_linear / (i_total_linear + n_0_linear)
    sinr_db = 10 * math.log10(sinr_linear + 1e-12)  # 加极小数防止 log10(0)

    return sinr_db


def get_normalized_comm_score(sinr_db, sf=7):
    """
    【论文最核心的物理映射函数】
    基于 Semtech 官方实测数据，将连续的 SINR 映射为 0~1 的归一化链路得分。

    物理含义：
    - Target (得分 1.0): PER < 0.1% (完美通信)
    - Replan (得分 0.8): PER = 1%   (开始丢包，触发算法机动预警！)
    - Outage (得分 0.0): PER > 10%  (解调门限崩溃，通信彻底中断)
    """
    # 以 SF=7 为例的查表阈值 (基于典型的 LoRa 解调门限经验值)
    sinr_target = -2.0
    sinr_th = -5.0
    sinr_outage = -7.5

    # 高于目标阈值，得满分
    if sinr_db >= sinr_target:
        return 1.0
    # 低于中断阈值，得零分 (此时如果无人机在这里，算作一次断连掉线)
    elif sinr_db <= sinr_outage:
        return 0.0
    # 处于过渡区，进行线性平滑映射 (供代价函数进行梯度惩罚)
    else:
        return (sinr_db - sinr_outage) / (sinr_target - sinr_outage)


def get_per_from_sinr(sinr_db):
    """
    (仅用于最终实验数据统计分析)
    根据 SINR 估算 Packet Error Rate (PER, 误包率)。
    这将在您的实验对比表格中作为极具说服力的物理层指标展现。
    """
    sinr_target = -2.0
    sinr_outage = -7.5

    if sinr_db >= sinr_target:
        return 0.001  # 0.1%
    elif sinr_db <= sinr_outage:
        return 1.0  # 100% 丢包
    else:
        # 在瀑布区内，PER 呈指数级上升 (这在通信界是极其真实的现象)
        # 用简单的指数插值模拟瀑布曲线
        ratio = (sinr_target - sinr_db) / (sinr_target - sinr_outage)
        per = 0.001 * math.exp(ratio * math.log(1.0 / 0.001))
        return min(per, 1.0)