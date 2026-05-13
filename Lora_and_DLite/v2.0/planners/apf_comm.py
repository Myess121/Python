# planners/apf_comm.py
"""
通信感知人工势场法 (APF-Comm)
Baseline 算法：计算极快，但在城市/复杂干扰环境中极易陷入局部死锁
用于衬托 D* Lite 算法的全局完备性和路径最优性
"""
import math
from planners.base_planner import BasePlanner
from environment import calculate_physical_sinr, get_normalized_comm_score
from config import COMM_WEIGHT


class APFComm(BasePlanner):
    def __init__(self, s_start, s_goal, heuristic_type="euclidean"):
        super().__init__(s_start, s_goal, heuristic_type)
        self.k_att = 1.0  # 引力增益
        self.k_rep_comm = COMM_WEIGHT * 10  # 通信斥力增益 (需放大以确保能推开无人机)
        self.replan_score_th = 0.8  # 触发通信斥力的阈值
        self.max_steps = 1000  # 防止死锁导致无限循环

    def cost(self, s_start, s_goal, gs=None, jammers=None):
        pass  # APF 基于势场梯度移动，不使用传统的图搜索代价函数

    def get_potential(self, pos):
        """计算网格节点上的总势场 (引力 + 斥力)"""
        # 1. 引力势场 (吸引向终点)
        dist_to_goal = math.hypot(self.s_goal[0] - pos[0], self.s_goal[1] - pos[1])
        u_att = self.k_att * dist_to_goal

        # 2. 物理障碍物斥力 (直接设为无穷大，不可逾越)
        if self.grid_map.is_collision(pos):
            return float('inf')

        # 3. 通信斥力势场 (干扰源产生的推力)
        u_rep_comm = 0.0
        if self.gs is not None and self.jammers is not None:
            sinr = calculate_physical_sinr(pos, self.gs, self.jammers)
            comm_score = get_normalized_comm_score(sinr)

            # 只有当链路质量变差时，才产生通信斥力
            if comm_score < self.replan_score_th:
                u_rep_comm = self.k_rep_comm * (self.replan_score_th - comm_score)

        return u_att + u_rep_comm

    def plan(self, grid_map, gs=None, jammers=None):
        self.grid_map = grid_map
        self.gs = gs
        self.jammers = jammers
        self.path = [self.s_start]
        current = self.s_start

        moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        for _ in range(self.max_steps):
            if current == self.s_goal:
                return self.path

            best_next = None
            min_u = float('inf')

            # 评估周围8个邻居，向势场最低（引力最大、斥力最小）的方向移动
            for move in moves:
                neighbor = (current[0] + move[0], current[1] + move[1])

                # 简易防死锁：不走刚才走过的老路
                if neighbor in self.path:
                    continue

                if not self.grid_map.is_collision(neighbor):
                    u = self.get_potential(neighbor)
                    if u < min_u:
                        min_u = u
                        best_next = neighbor

            # 如果周围所有路都走不通（或全被走过了），说明陷入了局部死锁
            if best_next is None:
                # 记录死锁状态，这在您的实验表格里就是“任务失败率”或“掉线率”
                print("⚠️ APF-Comm 陷入局部死锁 (Local Minima)，无法到达终点！")
                break

            current = best_next
            self.path.append(current)

        return self.path

    def dynamic_replan(self):
        """
        APF 的动态重规划就是从当前点继续算梯度。
        实际上只需要以上一个点作为起点，继续执行 plan 即可。
        """
        if self.path:
            self.s_start = self.path[-1]
        return self.plan(self.grid_map, self.gs, self.jammers)