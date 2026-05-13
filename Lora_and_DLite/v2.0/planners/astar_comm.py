# planners/astar_comm.py
"""
通信感知 A* 算法 (Ablation Study 专用)
带有基于 SINR-PER 映射的通信代价，但每次重规划都需要全局重算 O(N log N)
用于在实验中衬托 D* Lite 局部增量更新的超高实时性
"""
import math
from planners.astar import AStar
from environment import calculate_physical_sinr, get_normalized_comm_score
from config import COMM_WEIGHT


class AStarComm(AStar):
    def __init__(self, s_start, s_goal, heuristic_type="euclidean"):
        super().__init__(s_start, s_goal, heuristic_type)
        self.alpha = COMM_WEIGHT
        self.replan_score_th = 0.8  # 对应 SINR=-5.0dB (PER 1%)

    def cost(self, s_start, s_goal, gs=None, jammers=None):
        """融合几何与真实物理通信链路的联合代价函数"""
        if self.grid_map.is_collision(s_start) or self.grid_map.is_collision(s_goal):
            return float('inf')

        dist_cost = math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

        if self.gs is not None and self.jammers is not None:
            # 1. 算底层物理 SINR
            sinr = calculate_physical_sinr(s_goal, self.gs, self.jammers)
            # 2. 映射为归一化得分 (0~1)
            comm_score = get_normalized_comm_score(sinr)

            # 3. 线性代价惩罚
            if comm_score < self.replan_score_th:
                comm_cost = self.alpha * (self.replan_score_th - comm_score)
            else:
                comm_cost = 0.0
        else:
            comm_cost = 0.0

        return dist_cost + comm_cost

    def dynamic_replan(self):
        """
        A* 的重规划是极其愚蠢的全局重算！
        每次干扰源移动，都要清空所有状态从头搜一遍
        """
        return self.plan(self.grid_map, self.gs, self.jammers)