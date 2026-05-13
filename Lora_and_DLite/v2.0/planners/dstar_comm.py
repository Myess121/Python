# planners/dstar_comm.py
"""
通信感知 D* Lite 规划器 (顶刊大修版)
核心升级：
1. 彻底移除主观 LQM，代之以基于实测 LoRa 数据的物理 SINR-PER 映射得分。
2. 采用严格的线性惩罚机制，确保不破坏启发式搜索的一致性。
"""
import math
from planners.dstar_lite import DStarLite
from environment import calculate_physical_sinr, get_normalized_comm_score
from config import COMM_WEIGHT


class DStarComm(DStarLite):
    def __init__(self, s_start, s_goal, heuristic_type="euclidean"):
        super().__init__(s_start, s_goal, heuristic_type)
        # alpha 即为我们要在消融实验中测试灵敏度的通信权重
        self.alpha = COMM_WEIGHT
        # 重规划的归一化得分阈值：0.8 对应物理层的 PER = 1% (安全缓冲边界)
        self.replan_score_th = 0.8

    def cost(self, s_start, s_goal, gs=None, jammers=None):
        """
        融合几何与通信的联合代价函数
        严格保证 C_total = C_dist + alpha * C_comm >= C_dist，维持算法最优性边界
        """
        # 1. 障碍物碰撞检测（最高优先级）
        if self.grid_map.is_collision(s_start) or self.grid_map.is_collision(s_goal):
            return float("inf")

        # 2. 基础几何欧式距离代价 C_dist
        dist_cost = math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

        # 3. 动态通信代价 C_comm 计算
        if self.gs is not None and self.jammers is not None:
            # 步骤 A: 计算底层真实的物理 SINR（包含距离衰减、动态干扰、对数正态阴影衰落）
            sinr = calculate_physical_sinr(s_goal, self.gs, self.jammers)

            # 步骤 B: 查表映射为归一化通信得分 (区间 0~1，1.0代表极好，0.0代表完全中断)
            comm_score = get_normalized_comm_score(sinr)

            # 步骤 C: 线性代价惩罚（主动规避逻辑）
            # 只有当链路质量跌破安全缓冲（0.8）时，才开始施加空间惩罚
            if comm_score < self.replan_score_th:
                # 链路越差，惩罚值越大，迫使算法寻找更优路径
                comm_cost = self.alpha * (self.replan_score_th - comm_score)
            else:
                comm_cost = 0.0
        else:
            comm_cost = 0.0

        # 返回联合代价
        return dist_cost + comm_cost