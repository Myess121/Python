# planners/astar.py
"""
标准 A* 静态路径规划算法
作为 Baseline 对照组：完全无视通信干扰，仅追求几何距离最短
"""
import math
import heapq
from planners.base_planner import BasePlanner

class AStar(BasePlanner):
    def __init__(self, s_start, s_goal, heuristic_type="euclidean"):
        super().__init__(s_start, s_goal, heuristic_type)
        self.moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def plan(self, grid_map, gs=None, jammers=None):
        self.grid_map = grid_map
        self.gs = gs
        self.jammers = jammers

        open_set = []
        # heapq 存储格式: (f_score, state)
        heapq.heappush(open_set, (0, self.s_start))
        came_from = {self.s_start: None}
        g_score = {self.s_start: 0}
        f_score = {self.s_start: self.heuristic(self.s_start)}
        closed_set = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == self.s_goal:
                return self._reconstruct_path(came_from, current)

            closed_set.add(current)

            for move in self.moves:
                neighbor = (current[0] + move[0], current[1] + move[1])
                # 碰到静态障碍物或超出边界则跳过
                if neighbor in closed_set or self.grid_map.is_collision(neighbor):
                    continue

                # 仅计算纯几何欧式距离代价
                tentative_g = g_score[current] + self.cost(current, neighbor, gs, jammers)

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # 无可行路径

    def cost(self, s_start, s_goal, gs=None, jammers=None):
        """A* 仅使用几何距离代价（忽略所有干扰和通信环境）"""
        if self.grid_map.is_collision(s_start) or self.grid_map.is_collision(s_goal):
            return float('inf')
        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def dynamic_replan(self):
        """
        标准 A* 没有任何增量更新能力！
        面对动态环境，它的重规划就是极其愚蠢的：清空一切，从头开始全局重算。
        """
        return self.plan(self.grid_map, self.gs, self.jammers)